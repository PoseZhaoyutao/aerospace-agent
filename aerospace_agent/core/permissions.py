"""权限系统 — 1:1 复刻 CCB types/permissions.ts。

工具执行前的权限检查：canUseTool 回调决定是否允许工具执行。

权限模式：
    - default: 正常模式，危险操作需要用户确认
    - plan: 计划模式，只读
    - bypassPermissions: 跳过所有权限检查
    - auto: 自动模式，基于安全分类器自动决策
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ======================================================================
# 权限模式
# ======================================================================

class PermissionMode:
    """权限模式常量。"""
    DEFAULT = "default"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"
    AUTO = "auto"


# ======================================================================
# 权限结果
# ======================================================================

@dataclass
class PermissionResult:
    """权限检查结果。

    behavior:
        - "allow": 允许执行
        - "deny": 拒绝执行
        - "ask": 需要用户确认
    """
    behavior: str  # "allow" | "deny" | "ask"
    updated_input: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


def allow_permission(input_data: Dict[str, Any]) -> PermissionResult:
    """快捷创建允许结果。"""
    return PermissionResult(behavior="allow", updated_input=input_data)


def deny_permission(message: str = "") -> PermissionResult:
    """快捷创建拒绝结果。"""
    return PermissionResult(behavior="deny", message=message)


# ======================================================================
# 权限规则
# ======================================================================

@dataclass
class ToolPermissionRule:
    """单条权限规则。"""
    tool_name: str
    pattern: str = "*"  # glob pattern for input matching
    source: str = "user"  # "user" | "project" | "settings"


@dataclass
class ToolPermissionContext:
    """工具权限上下文 — 贯穿整个会话。"""
    mode: str = PermissionMode.DEFAULT
    additional_working_directories: Dict[str, Any] = field(default_factory=dict)
    always_allow_rules: Dict[str, List[ToolPermissionRule]] = field(default_factory=dict)
    always_deny_rules: Dict[str, List[ToolPermissionRule]] = field(default_factory=dict)
    always_ask_rules: Dict[str, List[ToolPermissionRule]] = field(default_factory=dict)
    is_bypass_permissions_mode_available: bool = True
    is_auto_mode_available: bool = False
    should_avoid_permission_prompts: bool = False
    await_automated_checks_before_dialog: bool = False
    pre_plan_mode: Optional[str] = None


def get_empty_tool_permission_context() -> ToolPermissionContext:
    """创建空权限上下文。"""
    return ToolPermissionContext()


# ======================================================================
# CanUseTool 回调类型
# ======================================================================

# canUseTool 回调签名:
#   tool: Tool 对象
#   input: Dict[str, Any] — 工具输入
#   context: ToolUseContext — 执行上下文
#   assistant_message: AssistantMessage — 触发工具调用的消息
#   tool_use_id: str — 工具调用 ID
#   force_decision: bool — 是否强制决策（不询问用户）
# 返回: PermissionResult
CanUseToolFn = Callable[
    [Any, Dict[str, Any], Any, Any, str, bool],
    Any,  # 返回 PermissionResult 或 awaitable
]


# ======================================================================
# 默认权限检查器
# ======================================================================

async def default_can_use_tool(
    tool: Any,
    input_data: Dict[str, Any],
    context: Any,
    assistant_message: Any,
    tool_use_id: str,
    force_decision: bool = False,
) -> PermissionResult:
    """默认权限检查器 — 所有工具默认允许。

    实际项目中可替换为需要用户确认的版本。
    """
    return allow_permission(input_data)
