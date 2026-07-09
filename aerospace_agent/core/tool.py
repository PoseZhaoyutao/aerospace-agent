"""Tool 接口 — 1:1 复刻 CCB Tool.ts。

这是整个架构的核心：定义了工具的完整接口契约。

关键设计（照搬 CCB）：
    1. Tool 是一个完整的接口，不只是 call() — 还包括权限检查、并发安全、
       只读判定、输入验证、结果大小限制等
    2. buildTool() 工厂函数填充安全默认值（fail-closed）
    3. ToolUseContext 是工具执行时的完整上下文（消息历史、权限、中止信号等）
    4. ToolResult 包含数据 + 可选的新消息 + 上下文修改器
    5. maxResultSizeChars: 大输出自动写文件，避免上下文膨胀
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .messages import (
    AssistantMessage,
    AttachmentMessage,
    Message,
    SystemMessage,
    UserMessage,
    ToolResultBlock,
)
from .permissions import (
    CanUseToolFn,
    PermissionResult,
    ToolPermissionContext,
    allow_permission,
)


# ======================================================================
# 工具输入 Schema
# ======================================================================

@dataclass
class ToolParam:
    """工具参数定义。"""
    name: str
    type: str = "string"  # JSON Schema type
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolInputSchema:
    """工具输入 Schema — JSON Schema 格式。"""
    type: str = "object"
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    @classmethod
    def from_params(cls, params: List[ToolParam]) -> "ToolInputSchema":
        """从参数列表构建 schema。"""
        properties = {}
        required = []
        for p in params:
            prop: Dict[str, Any] = {"type": p.type, "description": p.description}
            if p.default is not None:
                prop["default"] = p.default
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return cls(properties=properties, required=required)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
        }

    def safe_parse(self, input_data: Dict[str, Any]) -> Any:
        """验证输入 — 返回 parsed result。"""
        # 简化版验证：检查 required 字段
        missing = [r for r in self.required if r not in input_data]
        if missing:
            return _ParseResult(success=False, error=f"缺少必填参数: {missing}")
        return _ParseResult(success=True, data=input_data)


@dataclass
class _ParseResult:
    success: bool
    data: Any = None
    error: str = ""


# ======================================================================
# 验证结果
# ======================================================================

@dataclass
class ValidationResult:
    """工具输入验证结果。"""
    result: bool
    message: str = ""
    error_code: int = 0


# ======================================================================
# 工具进度
# ======================================================================

@dataclass
class ToolProgressData:
    """工具进度数据基类。"""
    type: str = ""


@dataclass
class ToolProgress:
    """工具进度事件。"""
    tool_use_id: str
    data: ToolProgressData


ToolCallProgress = Callable[[ToolProgress], None]


# ======================================================================
# ToolResult
# ======================================================================

T = TypeVar("T")


@dataclass
class ToolResult:
    """工具执行结果。

    data: 工具返回的数据
    new_messages: 工具产生的附加消息（如进度、附件等）
    context_modifier: 修改 ToolUseContext 的函数（仅非并发安全工具）
    """
    data: Any = None
    new_messages: Optional[List[Message]] = None
    context_modifier: Optional[Callable[["ToolUseContext"], "ToolUseContext"]] = None
    mcp_meta: Optional[Dict[str, Any]] = None


# ======================================================================
# ToolUseContext
# ======================================================================

@dataclass
class ToolUseContext:
    """工具执行上下文 — 贯穿整个查询循环。

    包含工具执行所需的一切：消息历史、权限、中止信号、文件状态等。
    """
    # 选项
    options: "ToolUseContextOptions"
    # 中止控制器
    abort_controller: Any = None  # AbortController equivalent
    # 文件读取状态缓存
    read_file_state: Dict[str, Any] = field(default_factory=dict)
    # AppState 回调
    get_app_state: Optional[Callable[[], Any]] = None
    set_app_state: Optional[Callable[[Callable], Any]] = None
    # 消息历史
    messages: List[Message] = field(default_factory=list)
    # 进度跟踪
    set_in_progress_tool_use_ids: Optional[Callable] = None
    set_response_length: Optional[Callable] = None
    # 文件历史
    update_file_history_state: Optional[Callable] = None
    # 嵌套记忆路径（去重）
    loaded_nested_memory_paths: Any = None
    # 工具调用 ID
    tool_use_id: Optional[str] = None
    # 查询链追踪
    query_tracking: Optional[Dict[str, Any]] = None
    # 用户修改标志
    user_modified: bool = False


@dataclass
class ToolUseContextOptions:
    """ToolUseContext 的选项子对象。"""
    commands: List[Any] = field(default_factory=list)
    debug: bool = False
    main_loop_model: str = ""
    tools: List["Tool"] = field(default_factory=list)
    verbose: bool = False
    thinking_config: Any = None
    mcp_clients: List[Any] = field(default_factory=list)
    mcp_resources: Dict[str, Any] = field(default_factory=dict)
    is_non_interactive_session: bool = True
    custom_system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None
    max_budget_usd: Optional[float] = None


# ======================================================================
# Tool 接口
# ======================================================================

class Tool:
    """工具接口 — 1:1 复刻 CCB Tool 类型。

    每个工具必须实现以下方法：
        - call(): 执行工具
        - description(): 返回工具描述
        - input_schema: 输入参数 schema
        - is_enabled(): 是否启用
        - is_concurrency_safe(): 是否可并发执行
        - is_read_only(): 是否只读
        - check_permissions(): 权限检查
        - map_tool_result_to_block(): 将结果转为 ToolResultBlock
    """

    # 可选别名（向后兼容）
    aliases: List[str] = []

    # 搜索提示（3-10 词，用于 SearchExtraTools 关键词匹配）
    search_hint: Optional[str] = None

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def input_schema(self) -> ToolInputSchema:
        raise NotImplementedError

    @property
    def max_result_size_chars(self) -> int:
        """结果最大字符数，超过则写入文件。"""
        return 50000

    async def call(
        self,
        args: Dict[str, Any],
        context: ToolUseContext,
        can_use_tool: CanUseToolFn,
        parent_message: AssistantMessage,
        on_progress: Optional[ToolCallProgress] = None,
    ) -> ToolResult:
        """执行工具。"""
        raise NotImplementedError

    async def description(
        self,
        input_data: Dict[str, Any],
        options: Dict[str, Any],
    ) -> str:
        """返回工具描述。"""
        raise NotImplementedError

    def is_enabled(self) -> bool:
        """是否启用。"""
        return True

    def is_concurrency_safe(self, input_data: Dict[str, Any]) -> bool:
        """是否可并发执行（只读工具通常可以）。"""
        return False

    def is_read_only(self, input_data: Dict[str, Any]) -> bool:
        """是否只读。"""
        return False

    def is_destructive(self, input_data: Dict[str, Any]) -> bool:
        """是否不可逆（删除、覆盖、发送等）。"""
        return False

    def interrupt_behavior(self) -> str:
        """用户提交新消息时的行为：'cancel' 或 'block'。"""
        return "block"

    async def check_permissions(
        self,
        input_data: Dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """权限检查 — 默认允许。"""
        return allow_permission(input_data)

    async def validate_input(
        self,
        input_data: Dict[str, Any],
        context: ToolUseContext,
    ) -> ValidationResult:
        """输入验证。"""
        return ValidationResult(result=True)

    def get_path(self, input_data: Dict[str, Any]) -> Optional[str]:
        """工具操作的文件路径（如适用）。"""
        return None

    def user_facing_name(self, input_data: Optional[Dict[str, Any]] = None) -> str:
        """用户可见名称。"""
        return self.name

    def to_auto_classifier_input(self, input_data: Dict[str, Any]) -> Any:
        """安全分类器输入。"""
        return ""

    def map_tool_result_to_block(
        self,
        content: Any,
        tool_use_id: str,
    ) -> ToolResultBlock:
        """将工具结果转为 ToolResultBlock。"""
        if isinstance(content, str):
            result_content = content
        elif isinstance(content, dict):
            result_content = json.dumps(content, ensure_ascii=False, indent=2)
        elif isinstance(content, (list, tuple)):
            result_content = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            result_content = str(content)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=result_content,
            is_error=False,
        )

    def get_tool_use_summary(self, input_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """工具使用摘要（紧凑视图）。"""
        return None

    def get_activity_description(self, input_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """当前活动描述（spinner 显示）。"""
        return None


# ======================================================================
# buildTool 工厂 — 填充安全默认值
# ======================================================================

# 默认值（fail-closed）
_TOOL_DEFAULTS = {
    "is_enabled": lambda self: True,
    "is_concurrency_safe": lambda self, input_data: False,
    "is_read_only": lambda self, input_data: False,
    "is_destructive": lambda self, input_data: False,
    "check_permissions": lambda self, input_data, context: allow_permission(input_data),
    "to_auto_classifier_input": lambda self, input_data: "",
    "user_facing_name": lambda self, input_data=None: "",
}


def build_tool(tool_def: type) -> Tool:
    """从 ToolDef 构建 Tool 实例，填充安全默认值。

    类似 CCB 的 buildTool() — 确保所有工具有完整接口。

    Defaults (fail-closed):
        - is_enabled → True
        - is_concurrency_safe → False (假设不安全)
        - is_read_only → False (假设写操作)
        - is_destructive → False
        - check_permissions → allow (交给通用权限系统)
        - to_auto_classifier_input → '' (跳过分类器)
        - user_facing_name → name
    """
    instance = tool_def()

    # 填充默认值
    if not hasattr(instance, "is_enabled") or instance.is_enabled.__func__ is Tool.is_enabled:
        instance.is_enabled = lambda: True
    if not hasattr(instance, "is_concurrency_safe"):
        instance.is_concurrency_safe = lambda input_data: False
    if not hasattr(instance, "is_read_only"):
        instance.is_read_only = lambda input_data: False
    if not hasattr(instance, "is_destructive"):
        instance.is_destructive = lambda input_data: False

    # 设置 user_facing_name 默认值
    if not instance.user_facing_name(None):
        name = instance.name
        instance.user_facing_name = lambda input_data=None, _n=name: _n

    return instance


# ======================================================================
# 工具查找
# ======================================================================

def tool_matches_name(tool: Tool, name: str) -> bool:
    """检查工具是否匹配给定名称（主名称或别名）。"""
    return tool.name == name or name in getattr(tool, "aliases", [])


def find_tool_by_name(tools: List[Tool], name: str) -> Optional[Tool]:
    """按名称查找工具。"""
    for t in tools:
        if tool_matches_name(t, name):
            return t
    return None


# ======================================================================
# 文件状态缓存
# ======================================================================

@dataclass
class FileStateCache:
    """文件状态缓存 — 追踪文件读取状态。"""
    cache: Dict[str, Any] = field(default_factory=dict)

    def get(self, path: str) -> Any:
        return self.cache.get(path)

    def set(self, path: str, value: Any) -> None:
        self.cache[path] = value

    def has(self, path: str) -> bool:
        return path in self.cache

    def clone(self) -> "FileStateCache":
        return FileStateCache(cache=dict(self.cache))


def clone_file_state_cache(cache: FileStateCache) -> FileStateCache:
    """克隆文件状态缓存。"""
    return cache.clone()
