"""aerospace_agent.core 核心子包。

导出主编排器 ``AerospaceAgent`` 与默认装配工厂 ``create_default_agent``。
"""
from .agent import AerospaceAgent, create_default_agent
from .context_manager import ContextManager
from .llm_interface import LLMInterface, MockLLM, OpenAICompatibleLLM, create_llm
from .memory import LongTermMemory, ShortTermMemory

__all__ = [
    "AerospaceAgent",
    "create_default_agent",
    "ContextManager",
    "LLMInterface",
    "MockLLM",
    "OpenAICompatibleLLM",
    "create_llm",
    "LongTermMemory",
    "ShortTermMemory",
]
