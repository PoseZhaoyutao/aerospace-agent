"""Token 估算 — 1:1 复刻 CCB services/tokenEstimation.ts。

粗略 token 估算：不依赖 tokenizer，用字符数 / 3.5 近似。
用于 auto-compact 触发判断和大输出写文件决策。

关键设计（照搬 CCB）：
    1. rough_token_count(text) = ceil(len(text) / 3.5)
    2. estimate_message_tokens(messages) 遍历所有 content block 逐个估算
    3. tool_use block 估算 name + input JSON
    4. tool_result block 估算 content text
    5. thinking block 估算 thinking text
    6. image block 固定 2000 tokens
    7. 最终结果 × 4/3 向上取整（保守估计）
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List


# ======================================================================
# 常量
# ======================================================================

# 字符到 token 的转换比率（英文约 4:1，中文约 2:1，混合取 3.5:1）
CHARS_PER_TOKEN = 3.5

# 图片固定 token 数
IMAGE_MAX_TOKEN_SIZE = 2000

# 保守系数：估算值 × 4/3
CONSERVATIVE_RATIO = 4.0 / 3.0


# ======================================================================
# 粗略 token 估算
# ======================================================================

def rough_token_count(text: str) -> int:
    """粗略估算文本的 token 数。

    对应 CCB 的 roughTokenCountEstimation()。
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def rough_token_count_for_messages(messages: List[Any]) -> int:
    """粗略估算消息列表的 token 数。

    对应 CCB 的 roughTokenCountEstimationForMessages()。
    """
    total = 0
    for msg in messages:
        total += _estimate_single_message_tokens(msg)
    return math.ceil(total * CONSERVATIVE_RATIO)


def estimate_message_tokens(messages: List[Any]) -> int:
    """估算消息列表的 token 数（完整版）。

    对应 CCB 的 estimateMessageTokens()。

    遍历所有 user/assistant 消息的 content blocks，
    对每种 block 类型用不同策略估算。
    """
    total_tokens = 0

    for message in messages:
        # 获取消息类型
        msg_type = _get_msg_type(message)

        if msg_type not in ("user", "assistant"):
            continue

        content = _get_msg_content(message)
        if not isinstance(content, list):
            # 纯文本消息
            if isinstance(content, str):
                total_tokens += rough_token_count(content)
            continue

        for block in content:
            block_type = _get_block_type(block)

            if block_type == "text":
                total_tokens += rough_token_count(_get_block_text(block))
            elif block_type == "tool_result":
                total_tokens += _calculate_tool_result_tokens(block)
            elif block_type in ("image", "document"):
                total_tokens += IMAGE_MAX_TOKEN_SIZE
            elif block_type == "thinking":
                total_tokens += rough_token_count(_get_block_thinking(block))
            elif block_type == "tool_use":
                name = _get_block_name(block)
                input_data = _get_block_input(block) or {}
                total_tokens += rough_token_count(name + json.dumps(input_data, ensure_ascii=False))
            else:
                # 其他类型用 JSON 序列化估算
                total_tokens += rough_token_count(json.dumps(block, ensure_ascii=False, default=str))

    # 保守估计
    return math.ceil(total_tokens * CONSERVATIVE_RATIO)


# ======================================================================
# 工具结果 token 计算
# ======================================================================

def _calculate_tool_result_tokens(block: Any) -> int:
    """计算 tool_result block 的 token 数。

    对应 CCB 的 calculateToolResultTokens()。
    """
    content = _get_block_content(block)
    if not content:
        return 0

    if isinstance(content, str):
        return rough_token_count(content)

    if isinstance(content, list):
        total = 0
        for item in content:
            item_type = _get_block_type(item)
            if item_type == "text":
                total += rough_token_count(_get_block_text(item))
            elif item_type in ("image", "document"):
                total += IMAGE_MAX_TOKEN_SIZE
        return total

    return rough_token_count(str(content))


# ======================================================================
# 辅助函数 — 处理多种消息格式
# ======================================================================

def _get_msg_type(msg: Any) -> str:
    """获取消息类型。"""
    if hasattr(msg, "type"):
        return msg.type
    if isinstance(msg, dict):
        return msg.get("type", "")
    return ""


def _get_msg_content(msg: Any) -> Any:
    """获取消息内容。"""
    if hasattr(msg, "message") and msg.message:
        return msg.message.get("content")
    if isinstance(msg, dict):
        inner = msg.get("message", msg)
        if isinstance(inner, dict):
            return inner.get("content")
        return inner
    return None


def _get_block_type(block: Any) -> str:
    """获取 block 类型。"""
    if isinstance(block, dict):
        return block.get("type", "")
    if hasattr(block, "type"):
        return block.type
    return ""


def _get_block_text(block: Any) -> str:
    """获取 block 的文本内容。"""
    if isinstance(block, dict):
        return block.get("text", "")
    if hasattr(block, "text"):
        return block.text
    return ""


def _get_block_thinking(block: Any) -> str:
    """获取 thinking block 的内容。"""
    if isinstance(block, dict):
        return block.get("thinking", "")
    if hasattr(block, "thinking"):
        return block.thinking
    return ""


def _get_block_name(block: Any) -> str:
    """获取 tool_use block 的名称。"""
    if isinstance(block, dict):
        return block.get("name", "")
    if hasattr(block, "name"):
        return block.name
    return ""


def _get_block_input(block: Any) -> Any:
    """获取 tool_use block 的输入。"""
    if isinstance(block, dict):
        return block.get("input", {})
    if hasattr(block, "input"):
        return block.input
    return {}


def _get_block_content(block: Any) -> Any:
    """获取 block 的 content 字段。"""
    if isinstance(block, dict):
        return block.get("content")
    if hasattr(block, "content"):
        return block.content
    return None


def _estimate_single_message_tokens(msg: Any) -> int:
    """估算单条消息的 token 数。"""
    content = _get_msg_content(msg)
    if isinstance(content, str):
        return rough_token_count(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            block_type = _get_block_type(block)
            if block_type == "text":
                total += rough_token_count(_get_block_text(block))
            elif block_type == "tool_use":
                total += rough_token_count(_get_block_name(block) + json.dumps(_get_block_input(block), ensure_ascii=False))
            elif block_type == "tool_result":
                total += _calculate_tool_result_tokens(block)
            elif block_type == "thinking":
                total += rough_token_count(_get_block_thinking(block))
            elif block_type in ("image", "document"):
                total += IMAGE_MAX_TOKEN_SIZE
        return total
    return 0
