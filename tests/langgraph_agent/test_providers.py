from __future__ import annotations

import json

import httpx

from aerospace_agent.langgraph_agent.providers import (
    AnthropicClient,
    FallbackLLMClient,
    OpenAICompatibleClient,
    ProviderRegistry,
)


def test_openai_compatible_client_normalizes_chat_response() -> None:
    client = OpenAICompatibleClient(
        endpoint="https://llm.example.test/v1",
        model="demo",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )
        ),
    )

    result = client.chat_messages([{"role": "user", "content": "hello"}])

    assert result == {"content": "ok", "tool_calls": []}


def test_anthropic_client_maps_native_messages_response() -> None:
    client = AnthropicClient(
        endpoint="https://api.anthropic.test",
        model="claude-test",
        api_key="secret",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "anthropic ok"}]},
            )
        ),
    )

    result = client.chat_messages([{"role": "user", "content": "hello"}])

    assert result == {"content": "anthropic ok", "tool_calls": []}


def test_anthropic_client_maps_openai_tool_schema_to_native_tools() -> None:
    seen: dict[str, object] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "web.search",
                        "input": {"query": "orbit"},
                    }
                ]
            },
        )

    client = AnthropicClient(
        endpoint="https://api.anthropic.test",
        model="claude-test",
        api_key="secret",
        transport=httpx.MockTransport(transport),
    )
    result = client.chat_messages(
        [{"role": "user", "content": "search"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web.search",
                    "description": "Search public web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ],
    )

    assert seen["payload"]["tools"] == [
        {
            "name": "web.search",
            "description": "Search public web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    assert result["tool_calls"][0]["function"]["name"] == "web.search"


def test_registry_uses_next_provider_when_primary_is_unavailable() -> None:
    class Failing:
        def chat_messages(self, *_args, **_kwargs):
            raise OSError("primary down")

    class Working:
        def chat_messages(self, *_args, **_kwargs):
            return {"content": "fallback", "tool_calls": []}

    client = FallbackLLMClient([Failing(), Working()])

    assert client.chat_messages([{"role": "user", "content": "hello"}])["content"] == "fallback"


def test_registry_selects_configured_provider_name() -> None:
    registry = ProviderRegistry.from_configs(
        [
            {
                "name": "first",
                "kind": "openai_compatible",
                "endpoint": "https://first.example.test/v1",
                "model": "one",
                "enabled": True,
            },
            {
                "name": "second",
                "kind": "openai_compatible",
                "endpoint": "https://second.example.test/v1",
                "model": "two",
                "enabled": True,
            },
        ],
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
    )

    selected = registry.get("second")

    assert selected.model == "two"
    assert registry.names == ("first", "second")


def test_registry_orders_fallback_by_priority_then_name() -> None:
    registry = ProviderRegistry.from_configs(
        [
            {
                "name": "slow",
                "kind": "openai_compatible",
                "endpoint": "https://slow.example.test/v1",
                "model": "slow",
                "priority": 20,
            },
            {
                "name": "fast",
                "kind": "openai_compatible",
                "endpoint": "https://fast.example.test/v1",
                "model": "fast",
                "priority": 10,
            },
        ]
    )

    assert registry.names == ("fast", "slow")


def test_fallback_client_can_switch_primary_provider() -> None:
    registry = ProviderRegistry.from_configs(
        [
            {
                "name": "first",
                "kind": "openai_compatible",
                "endpoint": "https://first.example.test/v1",
                "model": "one",
            },
            {
                "name": "second",
                "kind": "openai_compatible",
                "endpoint": "https://second.example.test/v1",
                "model": "two",
            },
        ]
    )

    client = registry.fallback("first")
    client.select("second")

    assert client.model == "two"
    assert client.endpoint == "https://second.example.test/v1"


def test_provider_reads_api_key_only_from_declared_environment(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization", "")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok", "tool_calls": []}}]},
        )

    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    client = OpenAICompatibleClient(
        endpoint="https://llm.example.test/v1",
        model="demo",
        api_key_env="TEST_PROVIDER_KEY",
        transport=httpx.MockTransport(transport),
    )

    client.chat_messages([{"role": "user", "content": "hello"}])

    assert seen["authorization"] == "Bearer secret-value"
