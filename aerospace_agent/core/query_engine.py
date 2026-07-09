"""QueryEngine — 1:1 复刻 CCB QueryEngine.ts。

QueryEngine 拥有查询生命周期和会话状态。一次会话一个 QueryEngine。
每次 submitMessage() 在同一会话内开启新一轮。

关键设计（照搬 CCB）：
    1. 一个 QueryEngine 对应一个会话（conversation）
    2. submitMessage() 是主入口 — 异步生成器，yield SDKMessage
    3. 会话状态跨轮持久：messages、usage、file cache
    4. 系统提示词在每轮开始时重新组装
    5. 支持中断（interrupt）、最大轮数、预算限制
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, AsyncGenerator

from .context import (
    fetch_system_prompt_parts,
    as_system_prompt,
    append_system_context,
    get_user_context,
    get_system_context,
)
from .messages import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    ProgressMessage,
    RequestStartEvent,
    StreamEvent,
    SystemMessage,
    TombstoneMessage,
    ToolUseSummaryMessage,
    UserMessage,
    create_assistant_message,
    create_user_message,
    extract_tool_use_blocks,
    normalize_messages_for_api,
)
from .permissions import (
    CanUseToolFn,
    PermissionResult,
    ToolPermissionContext,
    default_can_use_tool,
    get_empty_tool_permission_context,
)
from .query import QueryParams, Terminal, is_result_successful, query
from .tool import (
    Tool,
    ToolUseContext,
    ToolUseContextOptions,
    FileStateCache,
    clone_file_state_cache,
    find_tool_by_name,
)


# ======================================================================
# 使用量追踪
# ======================================================================

@dataclass
class Usage:
    """API 使用量。"""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: Dict[str, int]) -> None:
        self.input_tokens += other.get("input_tokens", 0)
        self.output_tokens += other.get("output_tokens", 0)
        self.cache_read_input_tokens += other.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens += other.get("cache_creation_input_tokens", 0)
        self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "total_tokens": self.total_tokens,
        }


# ======================================================================
# QueryEngine 配置
# ======================================================================

@dataclass
class QueryEngineConfig:
    """QueryEngine 配置。"""
    cwd: str = ""
    tools: List[Tool] = field(default_factory=list)
    commands: List[Any] = field(default_factory=list)
    mcp_clients: List[Any] = field(default_factory=list)
    agents: List[Any] = field(default_factory=list)
    can_use_tool: CanUseToolFn = None
    initial_messages: Optional[List[Message]] = None
    read_file_cache: FileStateCache = field(default_factory=FileStateCache)
    custom_system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None
    user_specified_model: Optional[str] = None
    fallback_model: Optional[str] = None
    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    task_budget: Optional[Dict[str, int]] = None
    verbose: bool = False
    # LLM 接口
    llm: Any = None
    # 流式回调
    stream_callback: Optional[Callable[[str], None]] = None
    # 工具定义（OpenAI 格式）
    tools_def: Optional[List[Dict]] = None


# ======================================================================
# SDK 消息类型
# ======================================================================

@dataclass
class SDKMessage:
    """SDK 输出消息基类。"""
    type: str = "message"
    session_id: str = ""
    uuid: str = field(default_factory=lambda: str(_uuid.uuid4()))


@dataclass
class SDKResultMessage(SDKMessage):
    """查询结果消息。"""
    type: str = "result"
    subtype: str = "success"  # "success" | "error_max_turns" | "error_during_execution" | "error_max_budget_usd"
    is_error: bool = False
    duration_ms: int = 0
    num_turns: int = 0
    result: str = ""
    stop_reason: Optional[str] = None
    total_cost_usd: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class SDKAssistantMessage(SDKMessage):
    """助手消息 SDK 输出。"""
    type: str = "assistant"
    message: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SDKUserMessage(SDKMessage):
    """用户消息 SDK 输出。"""
    type: str = "user"
    message: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SDKSystemMessage(SDKMessage):
    """系统消息 SDK 输出。"""
    type: str = "system"
    subtype: str = ""


@dataclass
class SDKProgressMessage(SDKMessage):
    """进度消息 SDK 输出。"""
    type: str = "progress"
    tool_use_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# QueryEngine 类
# ======================================================================

class QueryEngine:
    """查询引擎 — 拥有查询生命周期和会话状态。

    对应 CCB 的 QueryEngine 类。

    一个 QueryEngine 对应一个会话。
    每次 submitMessage() 在同一会话内开启新一轮。
    状态（messages, usage, file cache）跨轮持久。
    """

    def __init__(self, config: QueryEngineConfig):
        self.config = config
        self.messages: List[Message] = list(config.initial_messages or [])
        self.permission_denials: List[Dict] = []
        self.total_usage = Usage()
        self.read_file_state: FileStateCache = config.read_file_cache
        self._abort_event = asyncio.Event() if asyncio.get_event_loop_policy() else None
        self._discovered_skill_names: set = set()
        self._session_id = str(_uuid.uuid4())
        # Token 追踪
        self._last_token_count = 0
        self._compact_count = 0

    # ------------------------------------------------------------------
    # submitMessage — 主入口
    # ------------------------------------------------------------------

    async def submit_message(
        self,
        prompt: str,
        options: Optional[Dict] = None,
    ) -> AsyncGenerator[SDKMessage, None]:
        """提交消息并流式返回结果。

        对应 CCB 的 submitMessage()。

        Args:
            prompt: 用户输入文本
            options: {uuid, is_meta}

        Yields:
            SDKMessage (assistant/user/system/progress/result)
        """
        start_time = time.time()
        prompt_uuid = (options or {}).get("uuid", str(_uuid.uuid4()))
        is_meta = (options or {}).get("is_meta", False)

        # 1. 添加用户消息
        user_msg = create_user_message(content=prompt, is_meta=is_meta)
        user_msg.uuid = prompt_uuid
        self.messages.append(user_msg)

        # 2. 组装系统提示词
        default_prompt, user_context, system_context = fetch_system_prompt_parts(
            tools=self.config.tools,
            custom_system_prompt=self.config.custom_system_prompt,
            append_system_prompt=self.config.append_system_prompt,
        )
        system_prompt = as_system_prompt(default_prompt)

        # 3. 构建 ToolUseContext
        tool_use_context = ToolUseContext(
            options=ToolUseContextOptions(
                commands=self.config.commands,
                debug=self.config.verbose,
                main_loop_model=self.config.user_specified_model or "",
                tools=self.config.tools,
                verbose=self.config.verbose,
                mcp_clients=self.config.mcp_clients,
                is_non_interactive_session=True,
                custom_system_prompt=self.config.custom_system_prompt,
                append_system_prompt=self.config.append_system_prompt,
                max_budget_usd=self.config.max_budget_usd,
            ),
            abort_controller=self._abort_event,
            read_file_state=self.read_file_state.cache,
            messages=self.messages,
            get_app_state=lambda: None,
            set_app_state=lambda f: None,
        )

        # 4. 构建 QueryParams
        params = QueryParams(
            messages=self.messages,
            system_prompt=system_prompt,
            user_context=user_context,
            system_context=system_context,
            can_use_tool=self.config.can_use_tool or default_can_use_tool,
            tool_use_context=tool_use_context,
            max_turns=self.config.max_turns,
            task_budget=self.config.task_budget,
            llm=self.config.llm,
            tools_def=self.config.tools_def,
            stream_callback=self.config.stream_callback,
        )

        # 5. 运行查询循环
        turn_count = 1
        last_stop_reason: Optional[str] = None
        last_assistant_msg: Optional[AssistantMessage] = None
        errors: List[str] = []

        async for message in query(params):
            if isinstance(message, Terminal):
                # 查询循环结束
                if message.reason == "completed":
                    pass
                elif message.reason == "max_turns":
                    yield SDKResultMessage(
                        session_id=self._session_id,
                        subtype="error_max_turns",
                        is_error=True,
                        duration_ms=int((time.time() - start_time) * 1000),
                        num_turns=turn_count,
                        stop_reason=last_stop_reason,
                        errors=[f"Reached maximum number of turns ({self.config.max_turns})"],
                    )
                    return
                elif message.reason == "model_error":
                    errors.append(str(message.error) if message.error else "Unknown model error")
                break

            elif isinstance(message, AssistantMessage):
                # 助手消息
                self.messages.append(message)
                last_assistant_msg = message
                if message.message:
                    last_stop_reason = message.message.get("stop_reason")
                    usage = message.message.get("usage", {})
                    if usage:
                        self.total_usage.add(usage)
                yield SDKAssistantMessage(
                    session_id=self._session_id,
                    message=message.message or {},
                )

            elif isinstance(message, UserMessage):
                # 工具结果消息
                self.messages.append(message)
                turn_count += 1
                yield SDKUserMessage(
                    session_id=self._session_id,
                    message=message.message or {},
                )

            elif isinstance(message, SystemMessage):
                # 系统消息（含 compact_boundary）
                if message.subtype == "compact_boundary":
                    self._compact_count += 1
                    # compact 后的消息列表已经被 query loop 替换
                    # 这里不追加到 self.messages（query loop 已处理）
                else:
                    self.messages.append(message)
                    if message.subtype == "api_error":
                        errors.append(str(message.content))
                yield SDKSystemMessage(
                    session_id=self._session_id,
                    subtype=message.subtype,
                )

            elif isinstance(message, ProgressMessage):
                # 进度消息
                yield SDKProgressMessage(
                    session_id=self._session_id,
                    tool_use_id=message.tool_use_id,
                    data=message.data,
                )

            elif isinstance(message, AttachmentMessage):
                # 附件消息
                attachment = message.attachment or {}
                if attachment.get("type") == "max_turns_reached":
                    yield SDKResultMessage(
                        session_id=self._session_id,
                        subtype="error_max_turns",
                        is_error=True,
                        duration_ms=int((time.time() - start_time) * 1000),
                        num_turns=attachment.get("turnCount", turn_count),
                        errors=[f"Reached maximum number of turns ({attachment.get('maxTurns')})"],
                    )
                    return

            elif isinstance(message, RequestStartEvent):
                pass  # 内部事件，不输出

        # 6. 提取最终结果
        result_text = ""
        is_error = False

        # 从所有助手消息中提取有意义的文本（从后往前找）
        # 跳过仅包含 [Tool Call: ...] 格式的消息（这是 normalize_messages_for_api 的产物）
        import re as _re
        _tool_call_pattern = _re.compile(r"^\s*\[Tool Call:.*\]\(.*\)\s*$", _re.DOTALL)

        if last_assistant_msg and last_assistant_msg.message:
            content = last_assistant_msg.message.get("content", [])
            if isinstance(content, list) and content:
                # 检查最后一个 block
                last_block = content[-1]
                if isinstance(last_block, dict) and last_block.get("type") == "text":
                    text = last_block.get("text", "")
                    # 如果文本只是规范化工具调用格式，跳过
                    if _tool_call_pattern.match(text):
                        result_text = ""
                    else:
                        result_text = text
                elif isinstance(last_block, dict) and last_block.get("type") == "tool_use":
                    # 最后一个 block 是 tool_use — 从之前的助手消息中找文本
                    result_text = ""

            # 如果最后一个助手消息没有有意义的文本，从之前的消息中找
            if not result_text:
                for msg in reversed(self.messages):
                    if isinstance(msg, AssistantMessage) and msg.message:
                        content = msg.message.get("content", [])
                        if isinstance(content, list) and content:
                            for block in reversed(content):
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text and not _tool_call_pattern.match(text):
                                        result_text = text
                                        break
                        if result_text:
                            break

            is_error = bool(last_assistant_msg.is_api_error_message)

        if errors and not result_text:
            is_error = True
            result_text = "\n".join(errors)

        # 7. yield 最终结果
        yield SDKResultMessage(
            session_id=self._session_id,
            subtype="success" if not is_error else "error_during_execution",
            is_error=is_error,
            duration_ms=int((time.time() - start_time) * 1000),
            num_turns=turn_count,
            result=result_text,
            stop_reason=last_stop_reason,
            usage=self.total_usage.to_dict(),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # 中断
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """中断当前查询。"""
        if self._abort_event:
            self._abort_event.set()

    def reset_abort(self) -> None:
        """重置中断状态，以便下一轮查询。"""
        self._abort_event = asyncio.Event()

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------

    def get_messages(self) -> List[Message]:
        """获取会话消息列表。"""
        return self.messages

    def get_read_file_state(self) -> FileStateCache:
        """获取文件状态缓存。"""
        return self.read_file_state

    def get_session_id(self) -> str:
        """获取会话 ID。"""
        return self._session_id

    def get_total_usage(self) -> Usage:
        """获取总使用量。"""
        return self.total_usage

    def set_model(self, model: str) -> None:
        """设置模型。"""
        self.config.user_specified_model = model

    def set_tools_def(self, tools_def: List[Dict]) -> None:
        """设置工具定义（OpenAI 格式）。"""
        self.config.tools_def = tools_def
