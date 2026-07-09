"""上下文压缩 — 1:1 复刻 CCB services/compact/ 目录。

三级压缩策略（照搬 CCB）：
    1. Microcompact：清除旧工具结果内容（替换为 "[Old tool result content cleared]"）
       - 只压缩 COMPACTABLE_TOOLS 中的工具结果
       - 保留最近 N 轮的工具结果不压缩
       - 不调用 LLM，纯本地操作
    2. Auto-compact：当 token 数超过阈值时，调用 LLM 生成对话摘要
       - 用小模型生成摘要
       - 插入 compact_boundary 标记
       - 摘要之后的消息保留，之前的被替换
    3. Partial compact：prompt-too-long 时的最后手段
       - 按 API round 分组丢弃最旧的消息组
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .messages import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    SystemMessage,
    UserMessage,
    ToolResultBlock,
    ToolUseBlock,
    create_assistant_message,
    create_user_message,
    extract_tool_use_blocks,
    get_messages_after_compact_boundary,
    normalize_messages_for_api,
)
from .token_estimation import estimate_message_tokens, rough_token_count

_logger = logging.getLogger(__name__)


# ======================================================================
# 常量
# ======================================================================

# 可压缩的工具名（只压缩这些工具的结果）
COMPACTABLE_TOOLS = {
    "read_file", "write_file", "edit_file", "list_directory",
    "search_files", "grep", "glob", "execute_command",
    "web_search", "web_fetch", "shell",
    "file_read", "file_write", "file_edit",
    # 航天工具
    "orbit_calculator", "orbital_velocity", "calculator",
}

# Microcompact 清除标记
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"

# Auto-compact 阈值
AUTO_COMPACT_THRESHOLD = 100_000  # tokens
AUTO_COMPACT_WARNING_THRESHOLD = 80_000  # tokens

# Compact 输出 token 限制
COMPACT_MAX_OUTPUT_TOKENS = 4096

# Post-compact 恢复文件数
POST_COMPACT_MAX_FILES_TO_RESTORE = 5
POST_COMPACT_TOKEN_BUDGET = 50_000
POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000


# ======================================================================
# Microcompact
# ======================================================================

@dataclass
class MicrocompactResult:
    """Microcompact 结果。"""
    messages: List[Message]
    cleared_tool_use_ids: List[str] = field(default_factory=list)


def microcompact_messages(
    messages: List[Message],
    keep_recent_turns: int = 3,
) -> MicrocompactResult:
    """清除旧工具结果内容 — 不调用 LLM。

    对应 CCB 的 microcompactMessages()。

    策略：
    1. 找到所有 COMPACTABLE_TOOLS 的 tool_use_id
    2. 保留最近 keep_recent_turns 轮的工具结果
    3. 将更早的工具结果内容替换为清除标记
    """
    # 1. 收集所有可压缩的 tool_use_id
    compactable_ids = set()
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.message:
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name", "") in COMPACTABLE_TOOLS:
                            compactable_ids.add(block.get("id", ""))
                    elif isinstance(block, ToolUseBlock):
                        if block.name in COMPACTABLE_TOOLS:
                            compactable_ids.add(block.id)

    if not compactable_ids:
        return MicrocompactResult(messages=messages)

    # 2. 按 API round 分组，确定哪些轮次需要清除
    groups = _group_messages_by_api_round(messages)
    total_groups = len(groups)

    # 保留最近 keep_recent_turns 轮不清除
    clear_from_group = max(0, total_groups - keep_recent_turns)

    # 3. 清除旧工具结果
    cleared_ids = []
    new_messages = []

    for group_idx, group in enumerate(groups):
        if group_idx < clear_from_group:
            # 清除这组中的工具结果
            for msg in group:
                if isinstance(msg, UserMessage) and msg.message:
                    content = msg.message.get("content", [])
                    if isinstance(content, list):
                        new_content = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                tool_id = block.get("tool_use_id", "")
                                if tool_id in compactable_ids:
                                    new_content.append({
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": TIME_BASED_MC_CLEARED_MESSAGE,
                                        "is_error": False,
                                    })
                                    cleared_ids.append(tool_id)
                                else:
                                    new_content.append(block)
                            elif isinstance(block, ToolResultBlock):
                                if block.tool_use_id in compactable_ids:
                                    new_content.append(ToolResultBlock(
                                        tool_use_id=block.tool_use_id,
                                        content=TIME_BASED_MC_CLEARED_MESSAGE,
                                        is_error=False,
                                    ))
                                    cleared_ids.append(block.tool_use_id)
                                else:
                                    new_content.append(block)
                            else:
                                new_content.append(block)
                        new_msg = UserMessage(
                            message={**msg.message, "content": new_content},
                            uuid=msg.uuid,
                            timestamp=msg.timestamp,
                        )
                        new_messages.append(new_msg)
                    else:
                        new_messages.append(msg)
                else:
                    new_messages.append(msg)
        else:
            # 保留这组
            new_messages.extend(group)

    return MicrocompactResult(messages=new_messages, cleared_tool_use_ids=cleared_ids)


# ======================================================================
# API Round 分组
# ======================================================================

def _group_messages_by_api_round(messages: List[Message]) -> List[List[Message]]:
    """按 API round 分组消息。

    对应 CCB 的 groupMessagesByApiRound()。

    每个 assistant 消息开始一个新的 round（除非与前一个 assistant 消息
    来自同一个 API 响应）。
    """
    groups: List[List[Message]] = []
    current: List[Message] = []
    last_assistant_id: Optional[str] = None

    for msg in messages:
        msg_id = None
        if isinstance(msg, AssistantMessage) and msg.message:
            msg_id = msg.message.get("id")

        if (isinstance(msg, AssistantMessage)
                and msg_id is not None
                and msg_id != last_assistant_id
                and current):
            groups.append(current)
            current = [msg]
        else:
            current.append(msg)

        if isinstance(msg, AssistantMessage) and msg_id:
            last_assistant_id = msg_id

    if current:
        groups.append(current)
    return groups


# ======================================================================
# Auto-compact
# ======================================================================

@dataclass
class CompactionResult:
    """Compact 结果。"""
    boundary_marker: SystemMessage
    summary_messages: List[UserMessage]
    attachments: List[AttachmentMessage] = field(default_factory=list)
    messages_to_keep: Optional[List[Message]] = None
    pre_compact_token_count: int = 0
    post_compact_token_count: int = 0


def is_auto_compact_needed(
    messages: List[Message],
    threshold: int = AUTO_COMPACT_THRESHOLD,
) -> bool:
    """检查是否需要 auto-compact。

    对应 CCB 的 autoCompactIfNeeded() 的触发判断。
    """
    # 只检查 compact boundary 之后的消息
    relevant = get_messages_after_compact_boundary(messages)
    token_count = estimate_message_tokens(relevant)
    return token_count > threshold


def get_compact_warning_level(
    messages: List[Message],
    threshold: int = AUTO_COMPACT_THRESHOLD,
    warning_threshold: int = AUTO_COMPACT_WARNING_THRESHOLD,
) -> str:
    """获取 compact 警告级别。

    Returns:
        "none" | "warning" | "critical"
    """
    relevant = get_messages_after_compact_boundary(messages)
    token_count = estimate_message_tokens(relevant)

    if token_count > threshold:
        return "critical"
    if token_count > warning_threshold:
        return "warning"
    return "none"


async def compact_conversation(
    messages: List[Message],
    llm: Any,
    system_prompt: str = "",
    keep_recent_turns: int = 2,
) -> CompactionResult:
    """压缩对话 — 调用 LLM 生成摘要。

    对应 CCB 的 compactConversation()。

    流程：
    1. 先执行 microcompact（清除旧工具结果）
    2. 保留最近 keep_recent_turns 轮消息
    3. 用 LLM 对旧消息生成摘要
    4. 插入 compact_boundary + 摘要 + 保留的消息
    """
    # 1. Microcompact
    mc_result = microcompact_messages(messages, keep_recent_turns=keep_recent_turns)
    compacted_messages = mc_result.messages

    # 2. 分组，保留最近 N 轮
    groups = _group_messages_by_api_round(compacted_messages)
    if len(groups) <= keep_recent_turns:
        # 不够压缩
        return CompactionResult(
            boundary_marker=SystemMessage(subtype="compact_boundary"),
            summary_messages=[],
            messages_to_keep=messages,
            pre_compact_token_count=estimate_message_tokens(messages),
            post_compact_token_count=estimate_message_tokens(messages),
        )

    pre_count = estimate_message_tokens(messages)

    # 3. 分割：旧消息用于摘要，新消息保留
    old_groups = groups[:-keep_recent_turns]
    recent_groups = groups[-keep_recent_turns:]

    old_messages: List[Message] = []
    for g in old_groups:
        old_messages.extend(g)

    recent_messages: List[Message] = []
    for g in recent_groups:
        recent_messages.extend(g)

    # 4. 构建摘要请求
    summary_prompt = _build_compact_prompt(old_messages)

    # 5. 调用 LLM 生成摘要
    summary_text = ""
    try:
        api_messages = [
            {"role": "system", "content": "你是对话摘要助手。将对话历史压缩为简洁摘要，保留关键信息。"},
            {"role": "user", "content": summary_prompt},
        ]
        summary_text = llm.chat(api_messages, max_tokens=COMPACT_MAX_OUTPUT_TOKENS, timeout=60)
    except Exception as e:
        _logger.warning("Compact LLM 调用失败: %s", e)
        summary_text = f"[Compact failed: {e}]"

    # 6. 构建 compact boundary
    boundary = SystemMessage(
        subtype="compact_boundary",
        content="Conversation compacted",
    )

    # 7. 构建摘要消息
    summary_msg = create_user_message(
        content=f"[Conversation Summary]\n{summary_text}",
        is_meta=True,
    )

    post_count = estimate_message_tokens([summary_msg] + recent_messages)

    return CompactionResult(
        boundary_marker=boundary,
        summary_messages=[summary_msg],
        messages_to_keep=recent_messages,
        pre_compact_token_count=pre_count,
        post_compact_token_count=post_count,
    )


def build_post_compact_messages(result: CompactionResult) -> List[Message]:
    """构建 compact 后的消息列表。

    对应 CCB 的 buildPostCompactMessages()。

    顺序：boundary → summary → kept messages
    """
    messages: List[Message] = [result.boundary_marker]
    messages.extend(result.summary_messages)
    if result.messages_to_keep:
        messages.extend(result.messages_to_keep)
    messages.extend(result.attachments)
    return messages


# ======================================================================
# Compact prompt 构建
# ======================================================================

def _build_compact_prompt(messages: List[Message]) -> str:
    """构建摘要 prompt。"""
    # 将消息转为可读文本
    lines = []
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.message:
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            lines.append(f"Assistant: {block.get('text', '')[:500]}")
                        elif block.get("type") == "tool_use":
                            lines.append(f"  [Tool: {block.get('name', '')}]")
        elif isinstance(msg, UserMessage) and msg.message:
            content = msg.message.get("content", [])
            if isinstance(content, str):
                lines.append(f"User: {content[:500]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        text = block.get("content", "")
                        if isinstance(text, str):
                            lines.append(f"  [Result: {text[:200]}]")
        elif isinstance(msg, UserMessage) and msg.message:
            content = msg.message.get("content")
            if isinstance(content, str):
                lines.append(f"User: {content[:500]}")

    conversation_text = "\n".join(lines)

    return (
        "请将以下对话历史压缩为简洁摘要。保留：\n"
        "1. 用户的原始需求\n"
        "2. 已完成的关键操作和结果\n"
        "3. 待完成的任务\n"
        "4. 重要的文件路径和配置\n\n"
        f"对话历史:\n{conversation_text}\n\n"
        "摘要:"
    )


# ======================================================================
# Partial compact — prompt-too-long 最后手段
# ======================================================================

def truncate_head_for_ptl_retry(
    messages: List[Message],
    token_gap: int = 0,
) -> Optional[List[Message]]:
    """丢弃最旧的消息组以解决 prompt-too-long。

    对应 CCB 的 truncateHeadForPTLRetry()。
    """
    groups = _group_messages_by_api_round(messages)
    if len(groups) < 2:
        return None

    if token_gap > 0:
        # 按 token gap 计算需要丢弃多少组
        acc = 0
        drop_count = 0
        for g in groups:
            acc += estimate_message_tokens(g)
            drop_count += 1
            if acc >= token_gap:
                break
    else:
        # 默认丢弃 20%
        drop_count = max(1, len(groups) // 5)

    drop_count = min(drop_count, len(groups) - 1)
    if drop_count < 1:
        return None

    sliced = []
    for g in groups[drop_count:]:
        sliced.extend(g)

    # 如果第一条是 assistant，需要插入一个 user 标记
    if sliced and isinstance(sliced[0], AssistantMessage):
        marker = create_user_message(
            content="[earlier conversation truncated for compaction retry]",
            is_meta=True,
        )
        sliced.insert(0, marker)

    return sliced
