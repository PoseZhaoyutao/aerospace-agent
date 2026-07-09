"""Agent 查询循环 — 1:1 复刻 CCB query.ts。

这是整个架构的心脏：query() 异步生成器实现了 Agent 的核心循环。

循环流程（照搬 CCB）：
    while True:
        1. 构建 API 消息（过滤非对话消息、应用 compact）
        2. 调用 LLM（chat_with_tools）
        3. 解析响应 → 提取 tool_use blocks
        4. 如果有 tool_use → run_tools() → 添加结果到消息 → continue
        5. 如果没有 tool_use → terminal (completed)
        6. 错误处理 / max_turns / budget 检查

与 CCB 的关键区别：
    - Qwen3 API 服务器不支持 OpenAI tools 参数
    - 使用 <tool_call> 标签注入策略（chat_with_tools 已处理）
    - Python asyncio 替代 TypeScript async generator
"""
from __future__ import annotations

import asyncio
import json
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncGenerator

from .messages import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    ProgressMessage,
    RequestStartEvent,
    StreamEvent,
    SystemMessage,
    TombstoneMessage,
    ToolUseBlock,
    ToolUseSummaryMessage,
    UserMessage,
    create_assistant_message,
    create_attachment_message,
    create_user_message,
    extract_tool_use_blocks,
    get_messages_after_compact_boundary,
    normalize_messages_for_api,
)
from .permissions import CanUseToolFn
from .tool import Tool, ToolUseContext, ToolUseContextOptions, find_tool_by_name
from .tool_orchestration import run_tools, MessageUpdate


# ======================================================================
# 终端状态
# ======================================================================

@dataclass
class Terminal:
    """查询循环的终端状态。"""
    reason: str  # "completed" | "aborted_streaming" | "aborted_tools" | "model_error" | "max_turns"
    error: Optional[Any] = None


@dataclass
class Continue:
    """查询循环的继续状态。"""
    reason: str


# ======================================================================
# 查询参数
# ======================================================================

@dataclass
class QueryParams:
    """query() 的参数。"""
    messages: List[Message]
    system_prompt: str
    user_context: Dict[str, str]
    system_context: Dict[str, str]
    can_use_tool: CanUseToolFn
    tool_use_context: ToolUseContext
    fallback_model: Optional[str] = None
    query_source: str = "sdk"
    max_turns: Optional[int] = None
    task_budget: Optional[Dict[str, int]] = None
    # LLM 接口
    llm: Any = None
    # 工具定义（OpenAI 格式）
    tools_def: Optional[List[Dict]] = None
    # 流式回调
    stream_callback: Optional[Any] = None


# ======================================================================
# 循环状态
# ======================================================================

@dataclass
class QueryLoopState:
    """循环间传递的可变状态。"""
    messages: List[Message]
    tool_use_context: ToolUseContext
    turn_count: int = 1
    transition: Optional[Continue] = None
    max_output_tokens_recovery_count: int = 0


# ======================================================================
# query() — 异步生成器入口
# ======================================================================

async def query(
    params: QueryParams,
) -> AsyncGenerator:
    """Agent 查询循环 — 异步生成器。

    对应 CCB 的 query()。

    Yields:
        Message | StreamEvent | RequestStartEvent | Terminal

    Python 不支持 yield* 语法，直接委托给 _query_loop。
    """
    async for item in _query_loop(params):
        yield item


# ======================================================================
# _query_loop — 主循环
# ======================================================================

async def _query_loop(
    params: QueryParams,
) -> AsyncGenerator:
    """Agent 主循环 — while(True) 直到 terminal。

    对应 CCB 的 queryLoop()。
    """
    state = QueryLoopState(
        messages=params.messages,
        tool_use_context=params.tool_use_context,
    )

    # yield RequestStartEvent
    yield RequestStartEvent()

    while True:
        # 0. Auto-compact 检查 — 如果 token 数超过阈值，先压缩
        from .compact import is_auto_compact_needed, compact_conversation, build_post_compact_messages
        from .token_estimation import estimate_message_tokens

        if is_auto_compact_needed(state.messages):
            try:
                if params.llm:
                    compaction = await compact_conversation(
                        state.messages,
                        params.llm,
                        params.system_prompt,
                    )
                    state.messages = build_post_compact_messages(compaction)
                    state.tool_use_context.messages = state.messages
                    # yield compact boundary
                    yield compaction.boundary_marker
                    for sm in compaction.summary_messages:
                        yield sm
            except Exception as e:
                # compact 失败不中断循环
                pass

        # 1. 构建发送给 API 的消息
        messages_for_query = get_messages_after_compact_boundary(state.messages)

        # 过滤掉非对话消息，转为 API 格式
        api_messages = normalize_messages_for_api(messages_for_query)

        # 添加系统提示词 + 用户上下文
        system_prompt = params.system_prompt
        if params.system_context:
            from .context import append_system_context
            system_prompt = append_system_context(system_prompt, params.system_context)

        # 添加用户上下文到 system prompt
        if params.user_context:
            context_parts = []
            for key, value in params.user_context.items():
                context_parts.append(f"[{key}]\n{value}")
            system_prompt = system_prompt + "\n\n" + "\n\n".join(context_parts)

        # 确保 system 消息在开头
        if not api_messages or api_messages[0].get("role") != "system":
            api_messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            # 合并系统提示词
            api_messages[0]["content"] = system_prompt + "\n\n" + api_messages[0]["content"]

        # 2. 调用 LLM — 优先同步 chat_with_tools，回退流式
        _already_streamed = False  # 标记是否已在 LLM 调用阶段流式输出
        try:
            # 流式回调 — 逐 chunk 输出（仅 stream_chat_with_tools 路径使用）
            _stream_chunks = []
            def _on_chunk(chunk):
                if chunk.type == "text_delta" and chunk.text:
                    _stream_chunks.append(chunk.text)
                    if params.stream_callback:
                        params.stream_callback(chunk.text)

            if params.llm and hasattr(params.llm, "chat_with_tools"):
                # 同步 chat_with_tools（优先，兼容性最好）
                resp = params.llm.chat_with_tools(
                    api_messages,
                    params.tools_def or [],
                    max_tokens=4096,
                    timeout=120,
                )
                # 伪流式：将完整文本分段输出
                if params.stream_callback and resp.get("content"):
                    text = resp["content"]
                    chunk_size = 20
                    for i in range(0, len(text), chunk_size):
                        params.stream_callback(text[i:i+chunk_size])
                    _already_streamed = True
            elif params.llm and hasattr(params.llm, "stream_chat_with_tools"):
                # 流式回退（部分 vLLM 配置可能不兼容）
                resp = params.llm.stream_chat_with_tools(
                    api_messages,
                    params.tools_def or [],
                    on_chunk=_on_chunk,
                    max_tokens=4096,
                    timeout=120,
                )
                _already_streamed = bool(_stream_chunks)
            elif params.llm:
                # 回退到普通 chat
                text = params.llm.chat(api_messages, timeout=120)
                resp = {"content": text, "tool_calls": None, "finish_reason": "stop"}
                if params.stream_callback and text:
                    chunk_size = 20
                    for i in range(0, len(text), chunk_size):
                        params.stream_callback(text[i:i+chunk_size])
                    _already_streamed = True
            else:
                # 无 LLM，返回错误
                yield Terminal(reason="model_error", error="No LLM configured")
                return
        except Exception as e:
            # API 错误
            error_msg = f"LLM 调用失败: {e}"
            yield SystemMessage(subtype="api_error", content=error_msg)
            yield Terminal(reason="model_error", error=e)
            return

        # 3. 解析响应
        content = resp.get("content") or ""
        tool_calls = resp.get("tool_calls")
        finish_reason = resp.get("finish_reason", "stop")

        # 流式回调 — 仅在未流式输出过时补充（避免重复输出）
        if params.stream_callback and content and not _already_streamed:
            params.stream_callback(content)

        # 构建 AssistantMessage
        content_blocks = []
        if content:
            content_blocks.append({"type": "text", "text": content})
        if tool_calls:
            for tc in tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"call_{_uuid.uuid4().hex[:8]}"),
                    "name": tc.get("name", ""),
                    "input": tc.get("arguments", {}),
                })

        assistant_msg = create_assistant_message(
            content=content_blocks,
            stop_reason=finish_reason,
            usage=resp.get("usage", {}),
            model=getattr(params.llm, "model", None) if params.llm else None,
        )

        # yield assistant message
        yield assistant_msg
        state.messages.append(assistant_msg)

        # 更新 tool_use_context messages
        state.tool_use_context.messages = state.messages

        # 4. 检查是否有工具调用
        tool_use_blocks = []
        if tool_calls:
            for tc in tool_calls:
                tool_use_blocks.append(ToolUseBlock(
                    id=tc.get("id", f"call_{_uuid.uuid4().hex[:8]}"),
                    name=tc.get("name", ""),
                    input=tc.get("arguments", {}),
                ))

        if not tool_use_blocks:
            # 5. 没有工具调用
            # 检查是否有内容 — 如果内容为空，注入提示让 LLM 给出最终答案
            if not content and state.turn_count < (params.max_turns or 25):
                state.max_output_tokens_recovery_count += 1
                if state.max_output_tokens_recovery_count <= 2:
                    # 注入提示，让 LLM 重新回答
                    retry_msg = create_user_message(
                        "请根据以上工具执行结果，给出最终答案。直接用自然语言回答，不要调用工具。"
                    )
                    state.messages.append(retry_msg)
                    state.turn_count += 1
                    continue

            # terminal (completed)
            yield Terminal(reason="completed")
            return

        # 6. 执行工具
        try:
            async for update in run_tools(
                tool_use_blocks,
                [assistant_msg],
                params.can_use_tool,
                state.tool_use_context,
            ):
                if update.message:
                    yield update.message
                    state.messages.append(update.message)

                if update.new_context:
                    state.tool_use_context = update.new_context
                    state.tool_use_context.messages = state.messages

        except asyncio.CancelledError:
            yield Terminal(reason="aborted_tools")
            return
        except Exception as e:
            # 工具执行错误 — 将错误作为工具结果返回
            error_msg = f"工具执行异常: {e}"
            for block in tool_use_blocks:
                from .messages import ToolResultBlock
                error_result = create_user_message(
                    content=[ToolResultBlock(
                        tool_use_id=block.id,
                        content=error_msg,
                        is_error=True,
                    )],
                    tool_use_result=error_msg,
                    source_tool_assistant_uuid=assistant_msg.uuid,
                )
                yield error_result
                state.messages.append(error_result)

        # 7. 检查 max_turns
        state.turn_count += 1
        if params.max_turns and state.turn_count > params.max_turns:
            yield create_attachment_message({
                "type": "max_turns_reached",
                "turnCount": state.turn_count,
                "maxTurns": params.max_turns,
            })
            yield Terminal(reason="max_turns")
            return

        # 8. 检查 budget
        if params.task_budget:
            total = params.task_budget.get("total", 0)
            # 简化版预算检查
            total_tokens = sum(
                m.message.get("usage", {}).get("total_tokens", 0)
                for m in state.messages
                if isinstance(m, AssistantMessage) and m.message
            )
            if total and total_tokens >= total:
                yield Terminal(reason="max_turns", error="budget exceeded")
                return

        # continue loop
        state.transition = Continue(reason="tool_results_added")


# ======================================================================
# is_result_successful — 检查结果是否成功
# ======================================================================

def is_result_successful(
    result: Optional[Message],
    stop_reason: Optional[str],
) -> bool:
    """检查查询结果是否成功。

    对应 CCB 的 isResultSuccessful()。
    """
    if result is None:
        return False

    if isinstance(result, AssistantMessage):
        if result.message:
            content = result.message.get("content", [])
            if isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict):
                    return last_block.get("type") in ("text", "thinking")
                return True
            return stop_reason == "end_turn"
        return False

    if isinstance(result, UserMessage):
        # 工具结果消息也算成功
        return True

    return False


# ======================================================================
# yield* helper for Python
# ======================================================================

class _YieldFrom:
    """辅助类：模拟 TypeScript 的 yield* 语法。

    用法: yield* generator → for item in generator: yield item
    Python 中直接用 for 循环即可。
    """
    pass
