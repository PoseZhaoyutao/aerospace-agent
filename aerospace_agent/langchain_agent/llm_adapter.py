"""LLM 适配器 — 将现有 LLMInterface 包装为 langchain_core BaseChatModel。

Qwen3 通过 vLLM 提供 OpenAI 兼容 API，但不支持原生 function calling。
因此 BaseChatModel 的 bind_tools 路径不可用，我们通过 system prompt 注入工具定义，
用 ReActOutputParser 解析 Thought/Action 文本格式。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class LLMAdapter(BaseChatModel):
    """将 LLMInterface 包装为 BaseChatModel。

    Args:
        llm: 现有的 LLMInterface 实例（LocalLLM / MockLLM / ModelRouter）
        model_name: 模型标识名（用于日志/指标）
        max_tokens: 单次调用最大输出 token 数
        temperature: 采样温度
    """

    model_name: str = "qwen3"
    max_tokens: int = 1024
    temperature: float = 0.0

    # 内部持有 LLMInterface 引用（不参与 Pydantic 序列化）
    _llm: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm

    @property
    def _llm_type(self) -> str:
        return "aerospace-llm-adapter"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """将 LangChain BaseMessage 列表转为 OpenAI 格式 dict 列表。"""
        converted = []
        for m in messages:
            if isinstance(m, SystemMessage):
                converted.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, HumanMessage):
                converted.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                converted.append({"role": "assistant", "content": str(m.content)})
            elif isinstance(m, ToolMessage):
                converted.append({"role": "user", "content": f"Observation: {m.content}"})
            else:
                converted.append({"role": "user", "content": str(m.content)})
        return converted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """核心调用：messages → LLMInterface.chat() → AIMessage。"""
        converted = self._convert_messages(messages)

        # 调用底层 LLM
        try:
            response = self._llm.chat(converted)
        except Exception as e:
            # 向上抛，让 Agent 循环处理错误恢复
            raise RuntimeError(f"LLM 调用失败: {e}") from e

        # 处理流式返回（generator → 字符串）
        if hasattr(response, "__iter__") and not isinstance(response, str):
            chunks = []
            try:
                for chunk in response:
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    elif isinstance(chunk, dict):
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            chunks.append(content)
            except Exception:
                pass
            response = "".join(chunks)

        message = AIMessage(content=str(response) if response else "")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步暂不支持，同步回退。"""
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _stream_llm(self) -> Any:
        """返回底层 LLM 的 stream_chat 方法。"""
        return self._llm

    def stream_chat(self, messages: List[dict], **kwargs: Any) -> Iterator[str]:
        """流式调用底层 LLM。"""
        return self._llm.stream_chat(messages, **kwargs)