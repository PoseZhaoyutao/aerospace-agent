from __future__ import annotations

from aerospace_agent.langgraph_agent.runner import AgentRunner


class _Model:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat_messages(self, messages, **_kwargs):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "name": "file.read", "arguments": {"path": "README.md"}}
                ],
            }
        return {"content": "tool result summarized", "tool_calls": []}


def test_runner_executes_tool_and_feeds_result_back_to_model() -> None:
    model = _Model()
    runner = AgentRunner(
        model,
        tool_executor=lambda call: {"status": "success", "content": call.arguments["path"]},
    )

    result = runner.run([{"role": "user", "content": "read README"}])

    assert result.status == "success"
    assert result.content == "tool result summarized"
    assert result.iterations == 2
    assert result.tool_results == [{"status": "success", "content": "README.md"}]
    assert model.calls[1][-1]["role"] == "tool"
    assert "README.md" in model.calls[1][-1]["content"]


def test_runner_stops_before_model_when_cancelled() -> None:
    model = _Model()
    runner = AgentRunner(model, tool_executor=lambda _call: {})

    result = runner.run(
        [{"role": "user", "content": "hello"}],
        cancelled=lambda: True,
    )

    assert result.status == "interrupted"
    assert result.iterations == 0
    assert model.calls == []


def test_runner_returns_limit_reached_for_repeated_tool_loop() -> None:
    class LoopingModel:
        def chat_messages(self, messages, **_kwargs):
            return {
                "content": "",
                "tool_calls": [{"id": "loop", "name": "noop", "arguments": {}}],
            }

    runner = AgentRunner(
        LoopingModel(),
        tool_executor=lambda _call: {"status": "success"},
        max_iterations=2,
    )

    result = runner.run([{"role": "user", "content": "loop"}])

    assert result.status == "limit_reached"
    assert result.iterations == 2
