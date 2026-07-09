from pathlib import Path

from aerospace_agent.langchain_agent.basic_agent import (
    BasicLangChainAgent,
    build_basic_tools,
    write_text_file,
)


class ExplodingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        raise AssertionError("LLM should not be called for deterministic file tasks")


class OneShotLLM:
    def __init__(self, response="ok"):
        self.calls = 0
        self.response = response

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        return self.response


def _workspace(name: str) -> Path:
    root = Path(".test_runs") / "langchain_basic_agent" / name
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_write_text_file_rejects_path_escape():
    workspace = _workspace("reject_escape")

    result = write_text_file("../escape.html", "bad", workspace=workspace)

    assert result["status"] == "error"
    assert result["error_code"] == "PATH_OUTSIDE_WORKSPACE"
    assert not (workspace.parent / "escape.html").exists()


def test_write_text_file_writes_inside_workspace():
    workspace = _workspace("write_inside")

    result = write_text_file("demo/index.html", "hello", workspace=workspace)

    assert result["status"] == "ok"
    written = Path(result["path"])
    assert written == workspace / "demo" / "index.html"
    assert written.read_text(encoding="utf-8") == "hello"


def test_basic_agent_generates_static_site_without_llm_loop():
    workspace = _workspace("site")
    llm = ExplodingLLM()
    agent = BasicLangChainAgent(llm=llm, workspace=workspace)

    result = agent.invoke("写一个花里胡哨的网站保存到本地")

    assert result.ok is True
    assert result.action == "write_static_site"
    assert llm.calls == 0
    output_path = Path(result.artifacts[0])
    assert output_path.exists()
    assert "<html" in output_path.read_text(encoding="utf-8").lower()
    assert "reAct" not in result.output


def test_basic_agent_uses_single_llm_call_for_general_prompt():
    workspace = _workspace("oneshot")
    llm = OneShotLLM("two body answer")
    agent = BasicLangChainAgent(llm=llm, workspace=workspace)

    result = agent.invoke("解释两体问题")

    assert result.ok is True
    assert result.action == "llm_once"
    assert result.output == "two body answer"
    assert llm.calls == 1
    assert llm.messages[0]["role"] == "system"
    assert llm.messages[1]["role"] == "user"


def test_build_basic_tools_exposes_minimal_safe_tools():
    workspace = _workspace("tools")

    tools = build_basic_tools(workspace)

    assert [tool.name for tool in tools] == ["write_text_file", "list_basic_tools"]
