"""Provider-neutral bounded LLM/tool iteration loop.

`AgentRunner` does not resolve tools itself.  The injected executor must be an
authorized Agent Core executor (or a test double); this keeps the loop
independent from the side-effect policy.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunnerToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class RunnerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success", "interrupted", "limit_reached", "error"]
    content: str = ""
    iterations: int = 0
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return value


def _parse_response(raw: Any) -> tuple[str, list[RunnerToolCall]]:
    if isinstance(raw, str):
        return raw, []
    payload: Any = raw.model_dump(mode="python") if hasattr(raw, "model_dump") else raw
    if hasattr(payload, "content") and not isinstance(payload, Mapping):
        content = str(getattr(payload, "content", "") or "")
        calls = getattr(payload, "tool_calls", ()) or ()
        payload = {"content": content, "tool_calls": calls}
    if not isinstance(payload, Mapping):
        raise ValueError("model response must be text or a mapping")
    if isinstance(payload.get("choices"), list) and payload["choices"]:
        choice = payload["choices"][0]
        if isinstance(choice, Mapping):
            payload = choice.get("message", choice)
    if not isinstance(payload, Mapping):
        raise ValueError("model response message is invalid")
    content = payload.get("content", "")
    raw_calls = payload.get("tool_calls", ()) or ()
    calls: list[RunnerToolCall] = []
    for index, item in enumerate(raw_calls):
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = str(function.get("name", ""))
        if not name:
            raise ValueError("model tool call is missing a name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        if not isinstance(arguments, Mapping):
            raise ValueError("model tool call arguments must be an object")
        calls.append(
            RunnerToolCall(
                call_id=str(item.get("id") or f"call-{index}"),
                name=name,
                arguments=dict(arguments),
            )
        )
    return str(content or ""), calls


class AgentRunner:
    """Run a bounded model → tool calls → tool results loop."""

    def __init__(
        self,
        model: Any,
        *,
        tool_executor: Callable[[RunnerToolCall], Any],
        max_iterations: int = 50,
        max_tool_calls: int = 3,
        max_content_chars: int = 2_000,
        parallel_tools: bool = False,
    ) -> None:
        if model is None or not callable(getattr(model, "chat_messages", None)) and not callable(
            getattr(model, "chat", None)
        ):
            raise ValueError("model must expose chat_messages or chat")
        if not callable(tool_executor):
            raise TypeError("tool_executor must be callable")
        if max_iterations < 1 or max_tool_calls < 1 or max_content_chars < 1:
            raise ValueError("runner limits must be positive")
        self.model = model
        self.tool_executor = tool_executor
        self.max_iterations = int(max_iterations)
        self.max_tool_calls = int(max_tool_calls)
        self.max_content_chars = int(max_content_chars)
        self.parallel_tools = bool(parallel_tools)

    def _request(self, messages: Sequence[Mapping[str, Any]], system_prompt: str) -> Any:
        chat_messages = getattr(self.model, "chat_messages", None)
        if callable(chat_messages):
            return chat_messages(list(messages), system_prompt=system_prompt, stream=False)
        prompt = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
        )
        return self.model.chat(prompt, system_prompt=system_prompt, max_tokens=self.max_content_chars)

    def _execute_one(self, call: RunnerToolCall) -> dict[str, Any]:
        try:
            raw = self.tool_executor(call)
            value = _jsonable(raw)
            return dict(value) if isinstance(value, Mapping) else {"result": value}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        system_prompt: str = "",
        cancelled: Callable[[], bool] | None = None,
    ) -> RunnerResult:
        working = [dict(item) for item in messages]
        tool_results: list[dict[str, Any]] = []
        for iteration in range(1, self.max_iterations + 1):
            if cancelled is not None and cancelled():
                return RunnerResult(
                    status="interrupted",
                    iterations=iteration - 1,
                    tool_results=tool_results,
                    messages=working,
                )
            try:
                content, calls = _parse_response(self._request(working, system_prompt))
            except Exception as exc:
                return RunnerResult(
                    status="error",
                    iterations=iteration,
                    tool_results=tool_results,
                    messages=working,
                    error=str(exc),
                )
            content = content[: self.max_content_chars]
            working.append({"role": "assistant", "content": content, "tool_calls": [call.model_dump(mode="json") for call in calls]})
            if not calls:
                return RunnerResult(
                    status="success",
                    content=content,
                    iterations=iteration,
                    tool_results=tool_results,
                    messages=working,
                )
            if len(calls) > self.max_tool_calls:
                return RunnerResult(
                    status="error",
                    iterations=iteration,
                    tool_results=tool_results,
                    messages=working,
                    error="model requested too many tool calls in one iteration",
                )
            if self.parallel_tools and len(calls) > 1:
                with ThreadPoolExecutor(max_workers=self.max_tool_calls) as pool:
                    results = list(pool.map(self._execute_one, calls))
            else:
                results = [self._execute_one(call) for call in calls]
            for call, result in zip(calls, results):
                tool_results.append(result)
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
        return RunnerResult(
            status="limit_reached",
            iterations=self.max_iterations,
            tool_results=tool_results,
            messages=working,
            error="max_iterations exceeded",
        )


__all__ = ["AgentRunner", "RunnerResult", "RunnerToolCall"]
