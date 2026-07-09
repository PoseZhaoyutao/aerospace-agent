from aerospace_agent.core.agent import AerospaceAgent, Tool


class RepeatingActionLLM:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.text


class FailingStreamLLM:
    model = "failing-stream"

    def stream_chat(self, messages, **kwargs):
        raise RuntimeError("HTTP Error 400: Bad Request")
        yield ""  # pragma: no cover

    def chat(self, messages, **kwargs):
        raise RuntimeError("HTTP Error 400: Bad Request")


def test_parse_action_extracts_json_before_trailing_thought():
    text = (
        'Thought: choose a plot save action\n'
        'Action: save_figure\n'
        'Action Input: {"path": "demo_output/plot_request.png"}'
        'Thought: accidentally continued thinking'
    )

    action = AerospaceAgent._parse_action(text)

    assert action == {
        "tool": "save_figure",
        "args": {"path": "demo_output/plot_request.png"},
    }


def test_parse_action_uses_last_complete_action_block():
    text = (
        'Thought: old attempt\n'
        'Action: memory_recall\n'
        'Action Input: {"query": "plot"}\n'
        'Observation: tool unavailable\n'
        'Thought: new attempt\n'
        'Action: save_figure\n'
        'Action Input: {"path": "demo_output/plot_request.png"}\n'
    )

    action = AerospaceAgent._parse_action(text)

    assert action == {
        "tool": "save_figure",
        "args": {"path": "demo_output/plot_request.png"},
    }


def test_react_stream_stops_repeated_identical_tool_call():
    repeated = (
        'Thought: keep trying the same action\n'
        'Action: save_figure\n'
        'Action Input: {"path": "demo_output/plot_request.png"}'
    )
    llm = RepeatingActionLLM(repeated)
    tool_calls = []

    def save_figure(**kwargs):
        tool_calls.append(kwargs)
        return {"status": "success", "path": kwargs.get("path")}

    agent = AerospaceAgent(
        llm=llm,
        tools=[Tool("save_figure", "save a figure", save_figure)],
        max_steps=10,
    )

    result = agent.run_react_stream(
        "实现绘图",
        max_steps=10,
        stream_callback=None,
        enable_context=False,
    )

    assert "重复工具调用" in result
    assert llm.calls == 2
    assert len(tool_calls) == 1


def test_react_stream_returns_structured_llm_failure_instead_of_raising():
    agent = AerospaceAgent(llm=FailingStreamLLM(), tools=[], max_steps=3)

    result = agent.run_react_stream(
        "为什么失败",
        max_steps=3,
        stream_callback=lambda chunk: None,
        enable_context=False,
    )

    assert "LLM 调用失败" in result
    assert "HTTP Error 400" in result
