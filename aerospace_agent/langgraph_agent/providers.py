"""External and local model provider adapters.

The runtime speaks one small protocol (`chat`, `chat_messages`,
`stream_chat`). Providers are configured by name and API-key environment
variable; secrets are never serialized into graph state or configuration
snapshots.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class LLMUnavailableError(RuntimeError):
    """Raised when a provider cannot serve a request and fallback is allowed."""


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: Literal["openai_compatible", "anthropic"]
    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    enabled: bool = True
    priority: int = 0
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)


def _messages_with_system(messages: Sequence[Mapping[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    result = [dict(item) for item in messages]
    if system_prompt and not any(str(item.get("role", "")) == "system" for item in result):
        result.insert(0, {"role": "system", "content": system_prompt})
    return result


class OpenAICompatibleClient:
    """Client for OpenAI, Qwen, DeepSeek, Ollama, vLLM and compatible APIs."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else (
            os.environ.get(api_key_env, "") if api_key_env else ""
        )
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "zyt-agent/1.0"},
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_with_system(messages, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = [dict(item) for item in tools]
        try:
            response = self._client.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(f"{self.model} request failed: {exc}") from exc
        try:
            message = data["choices"][0]["message"]
            return {
                "content": str(message.get("content", "") or ""),
                "tool_calls": list(message.get("tool_calls", []) or []),
            }
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError("OpenAI-compatible response is invalid") from exc

    def chat(self, prompt: str, **kwargs: Any) -> str:
        result = self.chat_messages([{"role": "user", "content": prompt}], **kwargs)
        return str(result.get("content", ""))

    def stream_chat(self, prompt: str, **kwargs: Any):
        # Keep the compatibility contract even for providers where streaming
        # is not required by the graph; callers still receive a generator.
        yield self.chat(prompt, **kwargs)

    def is_available(self) -> bool:
        try:
            response = self._client.get(f"{self.endpoint}/models", headers=self._headers())
            return 200 <= response.status_code < 300
        except httpx.HTTPError:
            return False


class AnthropicClient:
    """Native Anthropic Messages API adapter."""

    def __init__(
        self,
        *,
        endpoint: str = "https://api.anthropic.com",
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else (
            os.environ.get(api_key_env, "") if api_key_env else ""
        )
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    def chat_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        system = system_prompt
        user_messages: list[dict[str, Any]] = []
        for item in messages:
            role = str(item.get("role", "user"))
            if role == "system":
                system = str(item.get("content", ""))
            else:
                user_messages.append({"role": role, "content": item.get("content", "")})
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system:
            payload["system"] = system
        if tools:
            native_tools: list[dict[str, Any]] = []
            for tool in tools:
                function = tool.get("function", tool)
                if not isinstance(function, Mapping):
                    continue
                name = str(function.get("name", "")).strip()
                if not name:
                    continue
                native_tools.append(
                    {
                        "name": name,
                        "description": str(function.get("description", "") or ""),
                        "input_schema": dict(function.get("parameters", {}) or {}),
                    }
                )
            if native_tools:
                payload["tools"] = native_tools
        try:
            response = self._client.post(f"{self.endpoint}/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(f"{self.model} request failed: {exc}") from exc
        blocks = data.get("content", []) if isinstance(data, Mapping) else []
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        tool_calls = [
            {
                "id": str(block.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(block.get("name", "")),
                    "arguments": block.get("input", {}),
                },
            }
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "tool_use"
        ]
        return {"content": text, "tool_calls": tool_calls}

    def chat(self, prompt: str, **kwargs: Any) -> str:
        result = self.chat_messages([{"role": "user", "content": prompt}], **kwargs)
        return str(result.get("content", ""))

    def stream_chat(self, prompt: str, **kwargs: Any):
        yield self.chat(prompt, **kwargs)

    def is_available(self) -> bool:
        return bool(self.api_key)


class FallbackLLMClient:
    """Try enabled providers in order, falling back only on availability errors."""

    def __init__(self, clients: Iterable[Any], *, named_clients: Mapping[str, Any] | None = None) -> None:
        self.clients = tuple(clients)
        if not self.clients:
            raise ValueError("at least one LLM provider is required")
        self.last_provider: Any | None = None
        self._named_clients = dict(named_clients or {})

    @property
    def model(self) -> str:
        return str(getattr(self.last_provider or self.clients[0], "model", ""))

    @property
    def endpoint(self) -> str:
        return str(getattr(self.last_provider or self.clients[0], "endpoint", ""))

    def select(self, name: str) -> None:
        """Make a configured provider primary while retaining safe fallback."""

        if name not in self._named_clients:
            raise KeyError(f"model provider is not configured: {name}")
        selected = self._named_clients[name]
        self.clients = (selected, *[item for item in self.clients if item is not selected])
        self.last_provider = selected

    def chat_messages(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        failures: list[str] = []
        for client in self.clients:
            try:
                result = client.chat_messages(messages, **kwargs)
                self.last_provider = client
                return result
            except (LLMUnavailableError, OSError, TimeoutError) as exc:
                failures.append(str(exc))
        raise LLMUnavailableError("all configured model providers failed: " + "; ".join(failures))

    def chat(self, prompt: str, **kwargs: Any) -> str:
        return str(self.chat_messages([{"role": "user", "content": prompt}], **kwargs).get("content", ""))

    def stream_chat(self, prompt: str, **kwargs: Any):
        yield self.chat(prompt, **kwargs)

    def is_available(self) -> bool:
        return any(bool(getattr(client, "is_available", lambda: True)()) for client in self.clients)


class ProviderRegistry:
    """Materialize strict provider configs and expose named clients."""

    def __init__(self, clients: Mapping[str, Any]) -> None:
        if not clients:
            raise ValueError("provider registry cannot be empty")
        self._clients = dict(clients)

    @classmethod
    def from_configs(
        cls,
        configs: Sequence[ProviderConfig | Mapping[str, Any]],
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> "ProviderRegistry":
        materialized: dict[str, Any] = {}
        normalized = sorted(
            [
                item
                if isinstance(item, ProviderConfig)
                else ProviderConfig.model_validate(item)
                for item in configs
            ],
            key=lambda item: (item.priority, item.name),
        )
        for config in normalized:
            if not config.enabled or config.name in materialized:
                continue
            client_type = OpenAICompatibleClient if config.kind == "openai_compatible" else AnthropicClient
            materialized[config.name] = client_type(
                endpoint=config.endpoint,
                model=config.model,
                api_key_env=config.api_key_env,
                timeout=config.timeout_seconds,
                transport=transport,
            )
        return cls(materialized)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._clients)

    def get(self, name: str) -> Any:
        try:
            return self._clients[str(name)]
        except KeyError as exc:
            raise KeyError(f"model provider is not configured: {name}") from exc

    def primary(self, name: str | None = None) -> Any:
        return self.get(name) if name else next(iter(self._clients.values()))

    def fallback(self, name: str | None = None) -> FallbackLLMClient:
        clients = list(self._clients.values())
        if name:
            selected = self.get(name)
            clients = [selected, *[item for item in clients if item is not selected]]
        return FallbackLLMClient(clients, named_clients=self._clients)


__all__ = [
    "AnthropicClient",
    "FallbackLLMClient",
    "LLMUnavailableError",
    "OpenAICompatibleClient",
    "ProviderConfig",
    "ProviderRegistry",
]
