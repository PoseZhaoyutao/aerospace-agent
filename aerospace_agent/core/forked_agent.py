"""Forked Agent — 1:1 复刻 CCB utils/forkedAgent.ts。

子 Agent：拥有独立上下文窗口的 Agent 实例。

关键设计（照搬 CCB）：
    1. Forked agent 共享父 Agent 的 cache-safe params（system prompt, tools, model）
       以利用 prompt cache
    2. 但拥有独立的消息列表（上下文窗口）
    3. 用途：compact 摘要生成、prompt suggestion、side question、
       speculation（先发制人执行可能需要的操作）
    4. 子 Agent 的使用量独立追踪
    5. 子 Agent 完成后返回 ForkedAgentResult（messages + usage + result text）
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .messages import (
    AssistantMessage,
    Message,
    UserMessage,
    create_user_message,
    extract_tool_use_blocks,
)
from .permissions import CanUseToolFn, default_can_use_tool
from .query import QueryParams, Terminal, query
from .tool import Tool, ToolUseContext, ToolUseContextOptions, FileStateCache, clone_file_state_cache

_logger = logging.getLogger(__name__)


# ======================================================================
# Cache-safe 参数（共享父 Agent 的缓存）
# ======================================================================

@dataclass
class CacheSafeParams:
    """与父 Agent 共享的缓存安全参数。

    这些参数必须与父 Agent 一致，才能命中 prompt cache：
    - system_prompt
    - user_context
    - system_context
    - tools
    - model
    """
    system_prompt: str
    user_context: Dict[str, str]
    system_context: Dict[str, str]
    tool_use_context: ToolUseContext
    fork_context_messages: List[Message] = field(default_factory=list)


# ======================================================================
# Forked Agent 参数
# ======================================================================

@dataclass
class ForkedAgentParams:
    """Forked Agent 参数。"""
    prompt_messages: List[Message]
    cache_safe_params: CacheSafeParams
    can_use_tool: CanUseToolFn = None
    query_source: str = "fork"
    fork_label: str = "fork"
    max_turns: Optional[int] = None
    max_output_tokens: Optional[int] = None
    on_message: Optional[Callable[[Message], None]] = None
    skip_transcript: bool = False
    skip_cache_write: bool = False


# ======================================================================
# Forked Agent 结果
# ======================================================================

@dataclass
class ForkedAgentResult:
    """Forked Agent 执行结果。"""
    messages: List[Message] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    result_text: str = ""
    error: Optional[str] = None
    num_turns: int = 0


# ======================================================================
# runForkedAgent — 执行子 Agent
# ======================================================================

async def run_forked_agent(
    params: ForkedAgentParams,
    llm: Any = None,
) -> ForkedAgentResult:
    """执行 forked agent 查询循环。

    对应 CCB 的 runForkedAgent()。

    流程：
    1. 克隆父 Agent 的 ToolUseContext（独立状态）
    2. 用 fork 的消息列表启动 query() 循环
    3. 收集所有消息和使用量
    4. 返回 ForkedAgentResult
    """
    result = ForkedAgentResult()

    # 1. 克隆 ToolUseContext（独立文件状态缓存）
    parent_ctx = params.cache_safe_params.tool_use_context
    fork_ctx = ToolUseContext(
        options=ToolUseContextOptions(
            commands=parent_ctx.options.commands,
            debug=parent_ctx.options.debug,
            main_loop_model=parent_ctx.options.main_loop_model,
            tools=parent_ctx.options.tools,
            verbose=parent_ctx.options.verbose,
            mcp_clients=parent_ctx.options.mcp_clients,
            mcp_resources=parent_ctx.options.mcp_resources,
            is_non_interactive_session=True,
            custom_system_prompt=parent_ctx.options.custom_system_prompt,
            append_system_prompt=parent_ctx.options.append_system_prompt,
            max_budget_usd=parent_ctx.options.max_budget_usd,
        ),
        abort_controller=parent_ctx.abort_controller,
        read_file_state=dict(parent_ctx.read_file_state),  # 克隆文件缓存
        messages=list(params.prompt_messages),
        get_app_state=parent_ctx.get_app_state,
        set_app_state=parent_ctx.set_app_state,
    )

    # 2. 构建 QueryParams
    query_params = QueryParams(
        messages=list(params.prompt_messages),
        system_prompt=params.cache_safe_params.system_prompt,
        user_context=params.cache_safe_params.user_context,
        system_context=params.cache_safe_params.system_context,
        can_use_tool=params.can_use_tool or default_can_use_tool,
        tool_use_context=fork_ctx,
        max_turns=params.max_turns,
        llm=llm,
        tools_def=_build_tools_def(parent_ctx.options.tools),
        stream_callback=None,
    )

    # 3. 运行查询循环
    turn_count = 1
    last_assistant_text = ""

    try:
        async for message in query(query_params):
            if isinstance(message, Terminal):
                if message.reason == "completed":
                    pass
                elif message.reason == "max_turns":
                    result.error = f"Reached max turns ({params.max_turns})"
                elif message.reason == "model_error":
                    result.error = str(message.error) if message.error else "Model error"
                break

            elif isinstance(message, AssistantMessage):
                result.messages.append(message)
                turn_count += 1

                # 提取文本
                if message.message:
                    content = message.message.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                last_assistant_text = block.get("text", "")

                    # 累积使用量
                    usage = message.message.get("usage", {})
                    for k, v in usage.items():
                        if isinstance(v, int):
                            result.usage[k] = result.usage.get(k, 0) + v

                if params.on_message:
                    params.on_message(message)

            elif isinstance(message, UserMessage):
                result.messages.append(message)
                turn_count += 1
                if params.on_message:
                    params.on_message(message)

    except Exception as e:
        result.error = str(e)
        _logger.warning("Forked agent error (%s): %s", params.fork_label, e)

    result.result_text = last_assistant_text
    result.num_turns = turn_count
    return result


# ======================================================================
# 辅助函数
# ======================================================================

def _build_tools_def(tools: List[Tool]) -> List[Dict]:
    """从 ToolInterface 列表构建 OpenAI 格式工具定义。"""
    from .tool_adapter import build_tools_def_for_query
    return build_tools_def_for_query(tools)


# ======================================================================
# 便捷函数 — 常见的 fork 场景
# ======================================================================

async def fork_for_summary(
    messages: List[Message],
    cache_safe_params: CacheSafeParams,
    llm: Any,
    max_turns: int = 1,
) -> str:
    """用 forked agent 生成对话摘要。

    用于 auto-compact。
    """
    prompt = create_user_message(
        content="请总结以上对话的关键信息，包括：用户需求、已完成操作、重要结果、待完成任务。",
    )
    params = ForkedAgentParams(
        prompt_messages=messages + [prompt],
        cache_safe_params=cache_safe_params,
        fork_label="compact_summary",
        max_turns=max_turns,
        skip_transcript=True,
        skip_cache_write=True,
    )
    result = await run_forked_agent(params, llm=llm)
    return result.result_text


async def fork_for_side_question(
    question: str,
    cache_safe_params: CacheSafeParams,
    llm: Any,
    context_messages: List[Message] = None,
    max_turns: int = 3,
) -> str:
    """用 forked agent 回答侧面问题。

    不干扰主对话，独立上下文。
    """
    messages = list(context_messages or [])
    messages.append(create_user_message(content=question))

    params = ForkedAgentParams(
        prompt_messages=messages,
        cache_safe_params=cache_safe_params,
        fork_label="side_question",
        max_turns=max_turns,
        skip_transcript=True,
    )
    result = await run_forked_agent(params, llm=llm)
    return result.result_text
