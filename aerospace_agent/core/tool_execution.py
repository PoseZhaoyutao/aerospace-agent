"""工具执行 — 1:1 复刻 CCB services/tools/toolExecution.ts。

单个工具的完整执行流程：
    1. validate_input() — 输入验证
    2. check_permissions() — 权限检查（canUseTool 回调）
    3. call() — 执行工具
    4. map_tool_result_to_block() — 结果转 ToolResultBlock
    5. 大输出写文件（maxResultSizeChars）
    6. 错误处理
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import traceback
from typing import Any, Dict, Optional

from .messages import (
    AssistantMessage,
    Message,
    ProgressMessage,
    UserMessage,
    ToolResultBlock,
    create_progress_message,
    create_user_message,
)
from .permissions import (
    CanUseToolFn,
    PermissionResult,
    allow_permission,
    deny_permission,
)
from .tool import (
    Tool,
    ToolResult,
    ToolUseContext,
    ValidationResult,
    find_tool_by_name,
)


# ======================================================================
# 大输出阈值
# ======================================================================

MAX_RESULT_SIZE_DEFAULT = 50000  # 字符


# ======================================================================
# runToolUse — 单个工具执行
# ======================================================================

async def run_tool_use(
    tool_use: Any,  # ToolUseBlock
    parent_message: Optional[AssistantMessage],
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> "AsyncGenerator[MessageUpdate, None]":
    """执行单个工具调用。

    对应 CCB 的 runToolUse()。

    流程：
    1. 查找工具
    2. 验证输入
    3. 权限检查
    4. 执行工具
    5. 处理结果（大输出写文件）
    6. 返回 ToolResultBlock 消息
    """
    from .tool_orchestration import MessageUpdate

    tool_use_id = getattr(tool_use, "id", "")
    tool_name = getattr(tool_use, "name", "")
    tool_input = getattr(tool_use, "input", {})
    if not isinstance(tool_input, dict):
        tool_input = {"value": tool_input}

    # 1. 查找工具
    tool = find_tool_by_name(tool_use_context.options.tools, tool_name)

    if tool is None:
        # 工具不存在
        error_msg = f"工具 '{tool_name}' 不存在。可用工具: {[t.name for t in tool_use_context.options.tools]}"
        result_block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )
        yield MessageUpdate(
            message=create_user_message(
                content=[result_block],
                tool_use_result=error_msg,
                source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
            ),
        )
        return

    # 2. 输入验证
    try:
        validation = await tool.validate_input(tool_input, tool_use_context)
        if not validation.result:
            error_msg = f"输入验证失败: {validation.message}"
            result_block = ToolResultBlock(
                tool_use_id=tool_use_id,
                content=error_msg,
                is_error=True,
            )
            yield MessageUpdate(
                message=create_user_message(
                    content=[result_block],
                    tool_use_result=error_msg,
                    source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
                ),
            )
            return
    except Exception as e:
        error_msg = f"输入验证异常: {e}"
        result_block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )
        yield MessageUpdate(
            message=create_user_message(
                content=[result_block],
                tool_use_result=error_msg,
                source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
            ),
        )
        return

    # 3. 权限检查
    try:
        # 先调用 canUseTool 回调
        perm_result = await _invoke_can_use_tool(
            can_use_tool,
            tool,
            tool_input,
            tool_use_context,
            parent_message,
            tool_use_id,
        )

        if perm_result.behavior == "deny":
            error_msg = f"权限拒绝: {perm_result.message or '操作不被允许'}"
            result_block = ToolResultBlock(
                tool_use_id=tool_use_id,
                content=error_msg,
                is_error=True,
            )
            yield MessageUpdate(
                message=create_user_message(
                    content=[result_block],
                    tool_use_result=error_msg,
                    source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
                ),
            )
            return

        if perm_result.behavior == "ask":
            # 在非交互模式下默认拒绝
            if tool_use_context.options.is_non_interactive_session:
                error_msg = f"权限拒绝（非交互模式不允许交互确认）: {perm_result.message or ''}"
                result_block = ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=error_msg,
                    is_error=True,
                )
                yield MessageUpdate(
                    message=create_user_message(
                        content=[result_block],
                        tool_use_result=error_msg,
                        source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
                    ),
                )
                return

        # 使用更新后的输入
        if perm_result.updated_input:
            tool_input = perm_result.updated_input

    except Exception as e:
        error_msg = f"权限检查异常: {e}"
        result_block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )
        yield MessageUpdate(
            message=create_user_message(
                content=[result_block],
                tool_use_result=error_msg,
                source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
            ),
        )
        return

    # 4. 执行工具
    try:
        # 发送进度消息
        activity = tool.get_activity_description(tool_input)
        if activity:
            yield MessageUpdate(
                message=create_progress_message(
                    tool_use_id=tool_use_id,
                    data={"type": "activity", "description": activity},
                ),
            )

        # 调用工具
        result = await tool.call(
            args=tool_input,
            context=tool_use_context,
            can_use_tool=can_use_tool,
            parent_message=parent_message or AssistantMessage(),
        )

        # 5. 处理结果
        result_content = result.data

        # 结果验证 — 检测错误/空结果，给 LLM 重试提示
        verification = _verify_tool_result(tool_name, result_content)
        if verification:
            result_content = result_content + "\n\n" + verification if result_content else verification

        # 大输出写文件
        max_size = getattr(tool, "max_result_size_chars", MAX_RESULT_SIZE_DEFAULT)
        if max_size and isinstance(result_content, str) and len(result_content) > max_size:
            tmp_path = _write_large_output_to_file(
                result_content,
                tool_use_id,
                tool_name,
            )
            preview = result_content[:500]
            result_content = (
                f"输出已写入文件: {tmp_path} (共 {len(result_content)} 字符)\n"
                f"预览:\n{preview}..."
            )

        # 转为 ToolResultBlock
        result_block = tool.map_tool_result_to_block(result_content, tool_use_id)

        # 创建工具结果消息
        result_message = create_user_message(
            content=[result_block],
            tool_use_result=result_content,
            source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
        )

        # 附加新消息（如果有）
        if result.new_messages:
            for msg in result.new_messages:
                yield MessageUpdate(message=msg)

        yield MessageUpdate(message=result_message)

        # 应用 context modifier（如果有）
        if result.context_modifier:
            yield MessageUpdate(
                new_context=result.context_modifier(tool_use_context),
            )

    except asyncio.CancelledError:
        # 工具被取消
        error_msg = "工具执行被用户取消"
        result_block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )
        yield MessageUpdate(
            message=create_user_message(
                content=[result_block],
                tool_use_result=error_msg,
                source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
            ),
        )

    except Exception as e:
        # 工具执行异常
        error_detail = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        error_msg = f"工具执行错误: {error_detail}\n\nTraceback:\n{tb[:2000]}"
        result_block = ToolResultBlock(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )
        yield MessageUpdate(
            message=create_user_message(
                content=[result_block],
                tool_use_result=error_msg,
                source_tool_assistant_uuid=parent_message.uuid if parent_message else None,
            ),
        )


# ======================================================================
# 辅助函数
# ======================================================================

async def _invoke_can_use_tool(
    can_use_tool: CanUseToolFn,
    tool: Tool,
    input_data: Dict[str, Any],
    context: ToolUseContext,
    assistant_message: Optional[AssistantMessage],
    tool_use_id: str,
    force_decision: bool = False,
) -> PermissionResult:
    """调用 canUseTool 回调，处理同步和异步两种情况。"""
    result = can_use_tool(
        tool,
        input_data,
        context,
        assistant_message or AssistantMessage(),
        tool_use_id,
        force_decision,
    )
    if asyncio.iscoroutine(result):
        result = await result
    if not isinstance(result, PermissionResult):
        # 兼容直接返回 bool 的情况
        if result:
            return allow_permission(input_data)
        else:
            return deny_permission("权限被拒绝")
    return result


def _write_large_output_to_file(
    content: str,
    tool_use_id: str,
    tool_name: str,
) -> str:
    """将大输出写入临时文件。

    对应 CCB 的 maxResultSizeChars 机制。
    """
    tmp_dir = tempfile.gettempdir()
    filename = f"tool_out_{tool_name}_{tool_use_id[:8]}.txt"
    filepath = os.path.join(tmp_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        # 写入失败则返回截断内容
        return f"(写入文件失败，返回截断内容)\n{content[:2000]}..."
    return filepath


# ======================================================================
# 结果验证 — 检测错误/空结果，给 LLM 重试提示
# ======================================================================

# 错误特征模式
_ERROR_PATTERNS = [
    "Traceback (most recent call last)",
    "Error:",
    "Exception:",
    "TypeError:",
    "ValueError:",
    "KeyError:",
    "AttributeError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "FileNotFoundError:",
    "PermissionError:",
    "RuntimeError:",
    "ZeroDivisionError:",
    "IndexError:",
    "NameError:",
    "is not defined",
    "No module named",
    "command not found",
    "syntax error",
    "connection refused",
    "HTTP 400",
    "HTTP 401",
    "HTTP 403",
    "HTTP 404",
    "HTTP 500",
    "HTTP 502",
    "HTTP 503",
]

# 空结果特征
_EMPTY_INDICATORS = [
    "",
    "None",
    "null",
    "[]",
    "{}",
    "No results",
    "No data",
    "空",
    "无结果",
    "未找到",
]


def _verify_tool_result(tool_name: str, result_content: Any) -> str:
    """验证工具执行结果，检测错误和空结果。

    如果检测到问题，返回给 LLM 的重试提示文本。
    如果结果正常，返回空字符串。

    Args:
        tool_name: 工具名称
        result_content: 工具返回的内容

    Returns:
        验证提示文本（有问题时）或空字符串（正常时）
    """
    if result_content is None:
        return (
            f"[验证提示] 工具 '{tool_name}' 返回了 None（空结果）。"
            f"可能原因：参数错误、数据不存在、或工具内部逻辑问题。"
            f"请检查输入参数是否正确，或尝试使用其他工具完成相同任务。"
        )

    # 转为字符串进行模式检测
    if not isinstance(result_content, str):
        try:
            content_str = json.dumps(result_content, ensure_ascii=False)
        except Exception:
            content_str = str(result_content)
    else:
        content_str = result_content

    content_stripped = content_str.strip()

    # 1. 空结果检测
    if content_stripped in _EMPTY_INDICATORS or len(content_stripped) == 0:
        return (
            f"[验证提示] 工具 '{tool_name}' 返回了空结果。"
            f"可能原因：查询条件不匹配、数据源为空、或参数有误。"
            f"请调整参数后重试，或换用其他工具。"
        )

    # 2. 错误模式检测
    detected_errors = []
    content_lower = content_str.lower()
    for pattern in _ERROR_PATTERNS:
        if pattern.lower() in content_lower:
            detected_errors.append(pattern)

    if detected_errors:
        error_list = ", ".join(detected_errors[:3])  # 最多列3个
        return (
            f"[验证提示] 工具 '{tool_name}' 的结果中检测到错误特征: {error_list}。"
            f"请分析错误原因并采取以下措施之一：\n"
            f"  1. 修正输入参数后重新调用此工具\n"
            f"  2. 换用其他可完成相同任务的工具\n"
            f"  3. 如果是依赖缺失（如 ModuleNotFoundError），"
            f"尝试用 create_tool 创建替代工具\n"
            f"  4. 如果无法解决，在最终答案中说明遇到的限制"
        )

    # 3. 结果过短检测（可能是无效返回）
    # 数字结果跳过长度检查（如 "1024", "3.14" 是有效的计算结果）
    try:
        float(content_stripped)
        return ""  # 是数字，跳过
    except ValueError:
        pass

    if len(content_stripped) < 3 and content_stripped not in ("OK", "ok", "N/A", "n/a"):
        return (
            f"[验证提示] 工具 '{tool_name}' 返回结果异常简短: '{content_stripped}'。"
            f"请确认这是预期的返回值，如果不是，请检查参数或换用工具。"
        )

    return ""
