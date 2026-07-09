"""aerospace_agent.core 核心子包。

导出主编排器 ``AerospaceAgent`` 与默认装配工厂 ``create_default_agent``，
以及上下文管理器、LLM 接口和三层记忆系统。

CCB 架构复刻模块（1:1 对应 Claude Code Best）：
    - messages.py      → 消息类型系统（对应 CCB types/message.ts）
    - permissions.py   → 权限系统（对应 CCB types/permissions.ts）
    - tool.py          → Tool 接口 + buildTool 工厂（对应 CCB Tool.ts）
    - context.py       → 系统提示词组装（对应 CCB context.ts）
    - tool_orchestration.py → 工具并发/串行编排（对应 CCB toolOrchestration.ts）
    - tool_execution.py     → 单工具执行流程（对应 CCB toolExecution.ts）
    - query.py         → Agent 查询循环（对应 CCB query.ts）
    - query_engine.py  → QueryEngine 会话管理（对应 CCB QueryEngine.ts）
    - tool_adapter.py  → 现有工具适配器

主要导出
--------
* :class:`AerospaceAgent`        Agent 主编排器（ReAct 循环 + QueryEngine）
* :func:`create_default_agent`   默认装配工厂
* :class:`QueryEngine`           CCB 架构查询引擎
* :class:`QueryEngineConfig`     查询引擎配置
* :class:`Tool`                  CCB 架构工具接口
* :func:`query`                  Agent 查询循环
"""
from .agent import AerospaceAgent, create_default_agent
from .context_manager import ContextManager
from .llm_interface import LLMInterface, MockLLM, OpenAICompatibleLLM, create_llm
from .memory import (
    ShortTermMemory,
    WorkingMemory,
    LongTermMemory,
    MemoryManager,
)
# CCB 架构复刻模块
from .messages import (
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ProgressMessage,
    ToolUseBlock,
    ToolResultBlock,
)
from .permissions import PermissionResult, PermissionMode
from .tool import Tool, ToolUseContext, ToolResult, build_tool
from .context import fetch_system_prompt_parts, as_system_prompt
from .tool_orchestration import run_tools, partition_tool_calls
from .tool_execution import run_tool_use
from .query import query, QueryParams, Terminal
from .query_engine import QueryEngine, QueryEngineConfig
from .tool_adapter import (
    CallableWrapperTool,
    ResearchToolAdapter,
    wrap_callable_tools,
    wrap_research_tools,
)
from .token_estimation import rough_token_count, estimate_message_tokens
from .compact import (
    microcompact_messages,
    compact_conversation,
    is_auto_compact_needed,
    get_compact_warning_level,
    build_post_compact_messages,
    truncate_head_for_ptl_retry,
)
from .forked_agent import (
    run_forked_agent,
    fork_for_summary,
    fork_for_side_question,
    ForkedAgentParams,
    ForkedAgentResult,
    CacheSafeParams,
)
from .streaming import stream_chat_with_tools, StreamChunk

__all__ = [
    # Agent
    "AerospaceAgent",
    "create_default_agent",
    # CCB 架构
    "QueryEngine",
    "QueryEngineConfig",
    "Tool",
    "ToolUseContext",
    "ToolResult",
    "build_tool",
    "query",
    "QueryParams",
    "Terminal",
    "run_tools",
    "partition_tool_calls",
    "run_tool_use",
    # 消息类型
    "AssistantMessage",
    "UserMessage",
    "SystemMessage",
    "ProgressMessage",
    "ToolUseBlock",
    "ToolResultBlock",
    # 权限
    "PermissionResult",
    "PermissionMode",
    # 上下文
    "fetch_system_prompt_parts",
    "as_system_prompt",
    # 工具适配器
    "CallableWrapperTool",
    "ResearchToolAdapter",
    "wrap_callable_tools",
    "wrap_research_tools",
    # Token 估算 + Compact
    "rough_token_count",
    "estimate_message_tokens",
    "microcompact_messages",
    "compact_conversation",
    "is_auto_compact_needed",
    "get_compact_warning_level",
    "build_post_compact_messages",
    "truncate_head_for_ptl_retry",
    # Forked Agent
    "run_forked_agent",
    "fork_for_summary",
    "fork_for_side_question",
    "ForkedAgentParams",
    "ForkedAgentResult",
    "CacheSafeParams",
    # Streaming
    "stream_chat_with_tools",
    "StreamChunk",
    # 上下文管理
    "ContextManager",
    # LLM
    "LLMInterface",
    "MockLLM",
    "OpenAICompatibleLLM",
    "create_llm",
    # 记忆系统
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    "MemoryManager",
]
