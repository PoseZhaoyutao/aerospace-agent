"""LangGraph 航天 Agent 模块。

基于 LangGraph 框架构建的航天领域专属 Agent，
集成上下文管理、MCP Server、RAG 知识库、循环检测、
断点续跑、技能自进化等全套能力。

主要导出:
    - LangGraphAerospaceAgent: Agent 主入口
    - create_agent: 工厂函数
    - AerospaceAgentState: 状态 TypedDict
    - AgentInput / AgentOutput: Pydantic I/O 协议
    - CycleDetector: 循环检测器
    - EvolutionEngine: 技能进化引擎
    - build_aerospace_graph: 图构建器
"""
from __future__ import annotations

from .agent import (
    LangGraphAerospaceAgent,
    SimpleLLMClient,
    create_agent,
)
from .state import AerospaceAgentState, create_initial_state
from .schema import (
    ActionType,
    AgentInput,
    AgentOutput,
    Decision,
    EvidenceItem,
    EvolutionFileChange,
    EvolutionProposal,
    EvolutionRecord,
    FrameType,
    TimeScale,
    ForceModel,
    PropagatorType,
    IntentType,
    RunStatus,
    OrbitState,
    KeplerianOrbitState,
    OrbitDesignRequest,
    OrbitDesignResponse,
    LaunchWindowRequest,
    LaunchWindowResponse,
    LunarTransferRequest,
    LunarTransferResponse,
    ToolCallRequest,
    ToolCallResponse,
    RagQueryRequest,
    RagQueryResponse,
    validate_input,
    validate_output,
    export_json_schemas,
)
from .cycle_detector import CycleDetector
from .evolution import EvolutionEngine, create_evolution_engine
from .graph import ServiceBundle, build_aerospace_graph, build_simple_graph, initial_input
from .checkpointer import (
    get_checkpointer,
    create_sqlite_checkpointer,
    create_memory_checkpointer,
    list_saved_threads,
    get_thread_checkpoints,
    delete_thread_checkpoints,
)
from .router import route_intent, classify_intent_keyword
from .safety import (
    ApprovalRequired,
    SafetyValidationError,
    SafetyValidator,
    orbital_specific_energy,
    requires_human_approval,
    two_body_acceleration,
    validate_input_payload,
    validate_orbital_payload,
    validate_tool_output,
)
from .providers import (
    AnthropicClient,
    FallbackLLMClient,
    OpenAICompatibleClient,
    ProviderConfig,
    ProviderRegistry,
)
from .runner import AgentRunner, RunnerResult, RunnerToolCall
from .turns import AgentLoop, CommandRouter, TurnContext, TurnState

__all__ = [
    # Agent
    "LangGraphAerospaceAgent",
    "create_agent",
    "SimpleLLMClient",
    # State
    "AerospaceAgentState",
    "create_initial_state",
    # Schema
    "RunStatus",
    "ActionType",
    "AgentInput",
    "AgentOutput",
    "Decision",
    "EvidenceItem",
    "EvolutionFileChange",
    "EvolutionProposal",
    "EvolutionRecord",
    "IntentType",
    "FrameType",
    "TimeScale",
    "ForceModel",
    "PropagatorType",
    "OrbitState",
    "KeplerianOrbitState",
    "OrbitDesignRequest",
    "OrbitDesignResponse",
    "LaunchWindowRequest",
    "LaunchWindowResponse",
    "LunarTransferRequest",
    "LunarTransferResponse",
    "ToolCallRequest",
    "ToolCallResponse",
    "RagQueryRequest",
    "RagQueryResponse",
    "validate_input",
    "validate_output",
    "export_json_schemas",
    # Cycle Detection
    "CycleDetector",
    # Evolution
    "EvolutionEngine",
    "create_evolution_engine",
    # Graph
    "build_aerospace_graph",
    "build_simple_graph",
    "ServiceBundle",
    "initial_input",
    # Checkpointer
    "get_checkpointer",
    "create_sqlite_checkpointer",
    "create_memory_checkpointer",
    "list_saved_threads",
    "get_thread_checkpoints",
    "delete_thread_checkpoints",
    # Router
    "route_intent",
    "classify_intent_keyword",
    # Safety validation
    "ApprovalRequired",
    "SafetyValidationError",
    "SafetyValidator",
    "orbital_specific_energy",
    "requires_human_approval",
    "two_body_acceleration",
    "validate_input_payload",
    "validate_orbital_payload",
    "validate_tool_output",
    # Turn lifecycle, runner, and model providers
    "AgentLoop",
    "AgentRunner",
    "AnthropicClient",
    "CommandRouter",
    "FallbackLLMClient",
    "OpenAICompatibleClient",
    "ProviderConfig",
    "ProviderRegistry",
    "RunnerResult",
    "RunnerToolCall",
    "TurnContext",
    "TurnState",
]

__version__ = "0.1.0"
