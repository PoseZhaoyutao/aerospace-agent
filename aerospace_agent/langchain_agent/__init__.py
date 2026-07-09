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
    SlidingWindowMemory,
    build_basic_tools,
    build_langchain_tools,
    create_basic_langchain_agent,
    extract_pdf_text,
    write_text_file,
)

try:
    import langchain_core  # noqa: F401
    LANGCHAIN_CORE_AVAILABLE = True
except ModuleNotFoundError:
    LANGCHAIN_CORE_AVAILABLE = False

__all__ = [
    "BasicAgentConfig",
    "BasicAgentResult",
    "BasicLangChainAgent",
    "BasicTool",
    "LANGCHAIN_CORE_AVAILABLE",
    "SlidingWindowMemory",
    "build_basic_tools",
    "build_langchain_tools",
    "create_basic_langchain_agent",
    "extract_pdf_text",
    "write_text_file",
]
