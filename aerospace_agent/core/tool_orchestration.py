"""工具编排 — 1:1 复刻 CCB services/tools/toolOrchestration.ts。

管理工具的并发/串行执行：
    1. partition_tool_calls() — 将工具调用分为批次
       - 连续的并发安全（只读）工具 → 并发批次
       - 非并发安全（写操作）工具 → 串行批次
    2. run_tools() — 异步生成器，按批次执行工具
    3. run_tools_concurrently() — 并发执行只读工具
    4. run_tools_serially() — 串行执行写操作工具
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .messages import (
    AssistantMessage,
    Message,
    UserMessage,
    ToolResultBlock,
    extract_tool_use_blocks,
)
from .permissions import CanUseToolFn
from .tool import (
    Tool,
    ToolResult,
    ToolUseContext,
    find_tool_by_name,
)


# ======================================================================
# 消息更新
# ======================================================================

@dataclass
class MessageUpdate:
    """工具执行产生的消息更新。"""
    message: Optional[Message] = None
    new_context: Optional[ToolUseContext] = None
    context_modifier: Optional[Dict[str, Any]] = None  # {tool_use_id, modify_context}


# ======================================================================
# 批次类型
# ======================================================================

@dataclass
class ToolBatch:
    """工具调用批次。"""
    is_concurrency_safe: bool
    blocks: List[Any]  # ToolUseBlock list


# ======================================================================
# 分区函数
# ======================================================================

def partition_tool_calls(
    tool_use_messages: List[Any],
    tool_use_context: ToolUseContext,
) -> List[ToolBatch]:
    """将工具调用分区为批次。

    规则（照搬 CCB）：
    1. 连续的并发安全工具 → 合并为一个并发批次
    2. 非并发安全工具 → 独立串行批次

    对应 CCB 的 partitionToolCalls()。
    """
    batches: List[ToolBatch] = []
    for tool_use in tool_use_messages:
        tool = find_tool_by_name(tool_use_context.options.tools, tool_use.name)
        is_safe = False
        if tool:
            try:
                is_safe = tool.is_concurrency_safe(tool_use.input)
            except Exception:
                is_safe = False

        if is_safe and batches and batches[-1].is_concurrency_safe:
            batches[-1].blocks.append(tool_use)
        else:
            batches.append(ToolBatch(
                is_concurrency_safe=is_safe,
                blocks=[tool_use],
            ))
    return batches


# ======================================================================
# 最大并发数
# ======================================================================

def get_max_tool_use_concurrency() -> int:
    """获取最大工具并发数。"""
    import os
    val = os.environ.get("AEROSPACE_MAX_TOOL_CONCURRENCY", "10")
    try:
        return int(val)
    except ValueError:
        return 10


# ======================================================================
# runTools — 工具执行编排主函数
# ======================================================================

async def run_tools(
    tool_use_messages: List[Any],
    assistant_messages: List[AssistantMessage],
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> "AsyncGenerator[MessageUpdate, None]":
    """执行工具调用 — 异步生成器。

    对应 CCB 的 runTools()。

    流程：
    1. partition_tool_calls() 分区
    2. 并发安全批次 → run_tools_concurrently()
    3. 非并发安全批次 → run_tools_serially()
    4. 收集 context_modifier 并在批次间应用
    """
    from .tool_execution import run_tool_use

    current_context = tool_use_context
    queued_context_modifiers: Dict[str, List] = {}

    for batch in partition_tool_calls(tool_use_messages, current_context):
        if batch.is_concurrency_safe:
            # 并发执行只读工具
            async for update in run_tools_concurrently(
                batch.blocks,
                assistant_messages,
                can_use_tool,
                current_context,
            ):
                if update.context_modifier:
                    tool_use_id = update.context_modifier.get("tool_use_id")
                    modifier = update.context_modifier.get("modify_context")
                    if tool_use_id and modifier:
                        if tool_use_id not in queued_context_modifiers:
                            queued_context_modifiers[tool_use_id] = []
                        queued_context_modifiers[tool_use_id].append(modifier)
                yield MessageUpdate(
                    message=update.message,
                    new_context=current_context,
                )

            # 应用排队的 context modifiers
            for block in batch.blocks:
                modifiers = queued_context_modifiers.get(getattr(block, "id", ""), [])
                for modifier in modifiers:
                    current_context = modifier(current_context)
            yield MessageUpdate(new_context=current_context)

        else:
            # 串行执行写操作工具
            async for update in run_tools_serially(
                batch.blocks,
                assistant_messages,
                can_use_tool,
                current_context,
            ):
                if update.new_context:
                    current_context = update.new_context
                yield MessageUpdate(
                    message=update.message,
                    new_context=current_context,
                )


# ======================================================================
# 并发执行
# ======================================================================

async def run_tools_concurrently(
    tool_use_messages: List[Any],
    assistant_messages: List[AssistantMessage],
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> "AsyncGenerator[MessageUpdate, None]":
    """并发执行多个只读工具。

    对应 CCB 的 runToolsConcurrently()。
    """
    from .tool_execution import run_tool_use

    max_concurrency = get_max_tool_use_concurrency()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(tool_use, parent_msg):
        async with semaphore:
            async for update in run_tool_use(
                tool_use,
                parent_msg,
                can_use_tool,
                tool_use_context,
            ):
                return update  # 只取第一个（结果消息）
        return None

    # 为每个工具找到对应的 parent assistant message
    tasks = []
    for tool_use in tool_use_messages:
        parent_msg = _find_parent_message(tool_use, assistant_messages)
        tasks.append(run_one(tool_use, parent_msg))

    # 并发执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            yield MessageUpdate(
                message=UserMessage(
                    message={"role": "user", "content": f"工具执行错误: {result}"},
                ),
            )
        elif result:
            yield result


# ======================================================================
# 串行执行
# ======================================================================

async def run_tools_serially(
    tool_use_messages: List[Any],
    assistant_messages: List[AssistantMessage],
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> "AsyncGenerator[MessageUpdate, None]":
    """串行执行多个写操作工具。

    对应 CCB 的 runToolsSerially()。
    """
    from .tool_execution import run_tool_use

    current_context = tool_use_context

    for tool_use in tool_use_messages:
        # 更新进行中的工具调用 ID
        if current_context.set_in_progress_tool_use_ids:
            tool_id = getattr(tool_use, "id", "")
            current_context.set_in_progress_tool_use_ids(
                lambda prev, _id=tool_id: prev | {_id} if isinstance(prev, set) else {_id}
            )

        parent_msg = _find_parent_message(tool_use, assistant_messages)

        async for update in run_tool_use(
            tool_use,
            parent_msg,
            can_use_tool,
            current_context,
        ):
            if update.new_context:
                current_context = update.new_context
            yield update

        # 清除进行中标记
        if current_context.set_in_progress_tool_use_ids:
            tool_id = getattr(tool_use, "id", "")
            current_context.set_in_progress_tool_use_ids(
                lambda prev, _id=tool_id: prev - {_id} if isinstance(prev, set) else set()
            )


# ======================================================================
# 辅助函数
# ======================================================================

def _find_parent_message(
    tool_use: Any,
    assistant_messages: List[AssistantMessage],
) -> Optional[AssistantMessage]:
    """找到包含给定 tool_use 的 AssistantMessage。"""
    for msg in assistant_messages:
        blocks = extract_tool_use_blocks(msg)
        for block in blocks:
            if block.id == getattr(tool_use, "id", ""):
                return msg
    return assistant_messages[-1] if assistant_messages else None
