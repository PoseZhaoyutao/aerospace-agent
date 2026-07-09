"""LangChain Agent 层 — 基于 langchain-core 的标准化 ReAct 循环。

替换核心 `core/agent.py` 中的手写 ReAct 循环，使用 langchain-core 的
``BaseTool``、``AgentAction``/``AgentFinish``、``BaseMessage`` 等标准化基类，
构建稳定、可测试、可扩展的 Agent 执行引擎。

核心组件:
    - LLMAdapter: 将现有 LLMInterface 包装为 BaseChatModel
    - ToolAdapter: 将现有 Tool 对象包装为 BaseTool
    - ReActOutputParser: 解析 Qwen3 的 Thought/Action/Final Answer 格式
    - ReActAgent: 标准 ReAct 循环（替代手写 run_react_stream/run）
"""
from .basic_agent import (
    BasicAgentConfig,
    BasicAgentResult,
    BasicLangChainAgent,
    BasicTool,
    build_basic_tools,
    build_langchain_tools,
    create_basic_langchain_agent,
    write_text_file,
)

LANGCHAIN_CORE_AVAILABLE = True
_LANGCHAIN_CORE_IMPORT_ERROR = None

try:
    from .llm_adapter import LLMAdapter
    from .tool_adapter import ToolAdapter, wrap_tools
    from .mcp_tool_adapter import MCPToolAdapter
    from .react_parser import ReActOutputParser, AgentAction, AgentFinish
    from .react_agent import ReActAgent, AgentConfig, AgentResult
except ModuleNotFoundError as exc:
    if exc.name != "langchain_core":
        raise
    LANGCHAIN_CORE_AVAILABLE = False
    _LANGCHAIN_CORE_IMPORT_ERROR = exc
    LLMAdapter = None
    ToolAdapter = None
    MCPToolAdapter = None
    wrap_tools = None
    ReActOutputParser = None
    AgentAction = None
    AgentFinish = None
    ReActAgent = None
    AgentConfig = None
    AgentResult = None

__all__ = [
    "BasicAgentConfig",
    "BasicAgentResult",
    "BasicLangChainAgent",
    "BasicTool",
    "LANGCHAIN_CORE_AVAILABLE",
    "build_basic_tools",
    "build_langchain_tools",
    "create_basic_langchain_agent",
    "write_text_file",
    "LLMAdapter",
    "ToolAdapter",
    "MCPToolAdapter",
    "wrap_tools",
    "ReActOutputParser",
    "AgentAction",
    "AgentFinish",
    "ReActAgent",
    "AgentConfig",
    "AgentResult",
]
