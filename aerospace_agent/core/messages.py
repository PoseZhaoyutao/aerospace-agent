"""消息类型系统 — 1:1 复刻 CCB types/message.ts。

所有对话中的消息都用这些类型表示。Agent 循环、工具执行、上下文管理
都围绕这些消息类型运作。

核心设计（照搬 CCB）：
    1. 消息有类型标签（assistant/user/system/progress/attachment/stream_event）
    2. AssistantMessage 的 content 是 ContentBlock 列表（text/tool_use/thinking）
    3. UserMessage 的 content 可以是 str 或 ContentBlock 列表（含 tool_result）
    4. 每条消息有 uuid 和 timestamp，用于追踪和持久化
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


# ======================================================================
# Content Block 类型
# ======================================================================

@dataclass
class TextBlock:
    """文本内容块。"""
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    """工具调用块 — LLM 请求调用工具。"""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    """工具结果块 — 工具执行结果返回给 LLM。"""
    type: str = "tool_result"
    tool_use_id: str = ""
    content: Any = None
    is_error: bool = False


@dataclass
class ThinkingBlock:
    """思考块 — LLM 的内部推理（extended thinking）。"""
    type: str = "thinking"
    thinking: str = ""


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock]


# ======================================================================
# 消息类型
# ======================================================================

@dataclass
class Message:
    """消息基类。"""
    type: str = "message"
    uuid: str = field(default_factory=lambda: str(_uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 以下是 CCB 中 Message 的可选字段
    message: Optional[Dict[str, Any]] = None  # API 格式的消息体
    tool_use_result: Any = None  # 原始工具结果对象
    is_meta: bool = False
    is_visible_in_transcript_only: bool = False
    is_api_error_message: bool = False
    api_error: Optional[str] = None
    is_compact_summary: bool = False
    source_tool_assistant_uuid: Optional[str] = None


@dataclass
class AssistantMessage(Message):
    """LLM 回复消息 — 包含 content blocks（text/tool_use/thinking）。"""
    type: str = "assistant"
    message: Optional[Dict[str, Any]] = None
    # message.content 是 ContentBlock 列表
    # message.stop_reason: "end_turn" | "tool_use" | "max_tokens" | None
    # message.usage: {input_tokens, output_tokens, cache_read_input_tokens, ...}


@dataclass
class UserMessage(Message):
    """用户消息或工具结果消息。

    工具结果以 UserMessage 形式存在，content 为 [ToolResultBlock(...)]。
    """
    type: str = "user"
    message: Optional[Dict[str, Any]] = None
    # content 可以是 str 或 ContentBlock 列表


@dataclass
class SystemMessage(Message):
    """系统消息 — compact_boundary / api_error / local_command 等。"""
    type: str = "system"
    subtype: str = ""  # "compact_boundary" | "api_error" | "local_command"
    content: Any = None
    compact_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProgressMessage(Message):
    """工具执行进度消息。"""
    type: str = "progress"
    tool_use_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttachmentMessage(Message):
    """附件消息 — structured_output / max_turns_reached / queued_command。"""
    type: str = "attachment"
    attachment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """流式事件 — message_start / message_delta / message_stop。"""
    type: str = "stream_event"
    event: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestStartEvent:
    """API 请求开始事件。"""
    type: str = "stream_request_start"


@dataclass
class TombstoneMessage(Message):
    """墓碑消息 — 用于移除消息的控制信号。"""
    type: str = "tombstone"


@dataclass
class ToolUseSummaryMessage(Message):
    """工具使用摘要消息。"""
    type: str = "tool_use_summary"
    summary: Any = None
    preceding_tool_use_ids: Any = None


# 所有消息类型的联合
AnyMessage = Union[
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ProgressMessage,
    AttachmentMessage,
    StreamEvent,
    RequestStartEvent,
    TombstoneMessage,
    ToolUseSummaryMessage,
    Message,
]


# ======================================================================
# 消息工厂函数
# ======================================================================

def create_user_message(
    content: Union[str, List[ContentBlock]],
    tool_use_result: Any = None,
    source_tool_assistant_uuid: Optional[str] = None,
    is_meta: bool = False,
) -> UserMessage:
    """创建用户消息。

    工具结果通过 content=[ToolResultBlock(...)] 传入。
    """
    msg = UserMessage(
        message={"role": "user", "content": content},
        tool_use_result=tool_use_result,
        source_tool_assistant_uuid=source_tool_assistant_uuid,
        is_meta=is_meta,
    )
    return msg


def create_assistant_message(
    content: List[ContentBlock],
    stop_reason: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
    model: Optional[str] = None,
) -> AssistantMessage:
    """创建 LLM 回复消息。"""
    msg = AssistantMessage(
        message={
            "role": "assistant",
            "content": content,
            "stop_reason": stop_reason,
            "usage": usage or {},
            "model": model,
        },
    )
    return msg


def create_system_message(
    subtype: str = "",
    content: Any = None,
) -> SystemMessage:
    """创建系统消息。"""
    return SystemMessage(subtype=subtype, content=content)


def create_progress_message(
    tool_use_id: str,
    data: Dict[str, Any],
) -> ProgressMessage:
    """创建进度消息。"""
    return ProgressMessage(tool_use_id=tool_use_id, data=data)


def create_attachment_message(
    attachment: Dict[str, Any],
) -> AttachmentMessage:
    """创建附件消息。"""
    return AttachmentMessage(attachment=attachment)


# ======================================================================
# 消息工具函数
# ======================================================================

def get_messages_after_compact_boundary(
    messages: List[Message],
) -> List[Message]:
    """获取 compact boundary 之后的消息。

    compact 后只保留 boundary 及之后的消息。
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, SystemMessage) and msg.subtype == "compact_boundary":
            return messages[i:]
    return messages


def normalize_messages_for_api(
    messages: List[Message],
) -> List[Dict[str, Any]]:
    """将消息列表规范化为 API 格式。

    过滤掉系统消息、进度消息等非对话消息，
    只保留 user/assistant 消息的 API 格式。

    重要：vLLM 不支持 OpenAI 的 tool_calls / tool role 消息格式。
    将 tool_use 和 tool_result 转为纯文本，让 LLM 通过上下文理解。
    """
    import json as _json
    result = []
    for msg in messages:
        if isinstance(msg, (AssistantMessage, UserMessage)):
            if not msg.message:
                continue
            role = msg.message.get("role", "user")
            content = msg.message.get("content")

            # 处理 content 为 list 的情况（含 tool_use / tool_result blocks）
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        bt = block.get("type", "")
                        if bt == "text":
                            text_parts.append(block.get("text", ""))
                        elif bt == "tool_use":
                            # 将 tool_use 转为文本（vLLM 不支持 tool_calls）
                            name = block.get("name", "")
                            inp = block.get("input", {})
                            text_parts.append(
                                f"[Tool Call: {name}]({_json.dumps(inp, ensure_ascii=False)})"
                            )
                        elif bt == "tool_result":
                            # 将 tool_result 转为文本
                            tid = block.get("tool_use_id", "")
                            tc = block.get("content", "")
                            is_err = block.get("is_error", False)
                            if isinstance(tc, (dict, list)):
                                tc = _json.dumps(tc, ensure_ascii=False)
                            prefix = "[Tool Error]" if is_err else "[Tool Result]"
                            text_parts.append(f"{prefix} {tid}: {tc}")
                        elif bt == "thinking":
                            # thinking block 跳过（不发给 API）
                            pass
                    elif isinstance(block, str):
                        text_parts.append(block)
                result.append({
                    "role": role,
                    "content": "\n".join(text_parts) if text_parts else "",
                })
            elif isinstance(content, str):
                result.append({"role": role, "content": content})
            else:
                result.append({"role": role, "content": str(content) if content else ""})
        elif isinstance(msg, Message) and msg.message:
            result.append(msg.message)
    return result


def count_tool_calls(messages: List[Message], tool_name: str) -> int:
    """统计消息中某工具的调用次数。"""
    count = 0
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.message:
            content = msg.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name") == tool_name:
                            count += 1
                    elif isinstance(block, ToolUseBlock) and block.name == tool_name:
                        count += 1
    return count


def extract_tool_use_blocks(message: AssistantMessage) -> List[ToolUseBlock]:
    """从 AssistantMessage 中提取所有 ToolUseBlock。"""
    if not message.message:
        return []
    content = message.message.get("content", [])
    if not isinstance(content, list):
        return []
    blocks = []
    for block in content:
        if isinstance(block, ToolUseBlock):
            blocks.append(block)
        elif isinstance(block, dict) and block.get("type") == "tool_use":
            blocks.append(ToolUseBlock(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input", {}),
            ))
    return blocks
