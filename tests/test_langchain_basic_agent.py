from pathlib import Path

from aerospace_agent.langchain_agent.basic_agent import (
    BasicAgentConfig,
    BasicLangChainAgent,
    SlidingWindowMemory,
    build_basic_tools,
    write_text_file,
)
from aerospace_agent.cli_tui import CEOEngine, Stats
from aerospace_agent.core.agent import AerospaceAgent
from aerospace_agent.skills.base import SkillBase
from aerospace_agent.skills.registry import SkillRegistry


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


class FakeRAG:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=3):
        self.calls.append((query, top_k))
        return [{"text": "RAG evidence alpha", "source": "fake"}]


class FakeMCPTool:
    name = "dummy"
    description = "fake mcp tool"
    source = "fallback"
    methods_schema = {"ping": {"params": ["value"]}}

    def list_methods(self):
        return ["ping"]

    def get_info(self):
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "methods": self.list_methods(),
        }

    def call(self, method, **kwargs):
        if method != "ping":
            return {"success": False, "error": "bad method"}
        return {
            "success": True,
            "source": self.source,
            "result": {"echo": kwargs},
            "error": None,
        }


class EchoSkill(SkillBase):
    name = "echo_skill"
    description = "Echoes input text."
    category = "test"

    def execute(self, agent, **kwargs):
        return {
            "success": True,
            "result": {"agent_seen": agent is not None, "echo": kwargs.get("text")},
            "message": "ok",
        }


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


def test_sliding_window_memory_keeps_recent_messages():
    memory = SlidingWindowMemory(max_messages=3, max_chars=1000)

    memory.add("user", "u1")
    memory.add("assistant", "a1")
    memory.add("user", "u2")
    memory.add("assistant", "a2")

    assert memory.to_messages() == [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_agent_injects_sliding_memory_into_llm_prompt():
    workspace = _workspace("memory")
    llm = OneShotLLM("memory answer")
    memory = SlidingWindowMemory(max_messages=2, max_chars=1000)
    memory.add("user", "old user")
    memory.add("assistant", "recent assistant")
    memory.add("user", "recent user")
    agent = BasicLangChainAgent(
        llm=llm,
        workspace=workspace,
        config=BasicAgentConfig(memory_window_messages=2),
        memory=memory,
    )

    result = agent.invoke("current question")

    contents = [message["content"] for message in llm.messages]
    assert result.output == "memory answer"
    assert "old user" not in contents
    assert "recent assistant" in contents
    assert "recent user" in contents
    assert contents[-1] == "current question"
    assert memory.messages[-2]["content"] == "current question"
    assert memory.messages[-1]["content"] == "memory answer"


def test_agent_injects_rag_context_when_available():
    workspace = _workspace("rag")
    llm = OneShotLLM("rag answer")
    rag = FakeRAG()
    agent = BasicLangChainAgent(llm=llm, workspace=workspace, rag=rag)

    result = agent.invoke("current question")

    assert result.output == "rag answer"
    assert rag.calls == [("current question", 3)]
    assert any("RAG evidence alpha" in message["content"] for message in llm.messages)


def test_build_basic_tools_exposes_minimal_safe_tools():
    workspace = _workspace("tools")

    tools = build_basic_tools(workspace)

    assert [tool.name for tool in tools] == [
        "write_text_file",
        "list_basic_tools",
        "list_mcp_tools",
        "call_mcp_tool",
        "list_skills",
        "use_skill",
        "discover_skill_manifests",
        "install_skill_from_path",
    ]


def test_basic_tools_can_call_registered_mcp_tool():
    workspace = _workspace("mcp")
    tools = build_basic_tools(workspace, mcp_tools={"dummy": FakeMCPTool()})
    call_tool = next(tool for tool in tools if tool.name == "call_mcp_tool")

    result = call_tool.invoke({
        "name": "dummy",
        "method": "ping",
        "arguments": {"value": 3},
    })

    assert result["status"] == "ok"
    assert result["result"]["success"] is True
    assert result["result"]["result"]["echo"] == {"value": 3}


def test_basic_tools_list_and_use_registered_python_skill():
    workspace = _workspace("skill_python")
    registry = SkillRegistry()
    registry.register(EchoSkill())
    agent_context = object()
    tools = build_basic_tools(
        workspace,
        skill_registry=registry,
        skill_agent=agent_context,
    )
    list_tool = next(tool for tool in tools if tool.name == "list_skills")
    use_tool = next(tool for tool in tools if tool.name == "use_skill")

    listed = list_tool.invoke()
    result = use_tool.invoke({
        "name": "echo_skill",
        "arguments": {"text": "hello"},
    })

    assert listed["status"] == "ok"
    assert listed["skills"][0]["name"] == "echo_skill"
    assert result["status"] == "ok"
    assert result["result"]["success"] is True
    assert result["result"]["result"] == {"agent_seen": True, "echo": "hello"}


def test_basic_tools_discover_skill_manifests():
    workspace = _workspace("skill_discover")
    skill_root = workspace / "skills"
    skill_dir = skill_root / "orbit-audit"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: orbit-audit
description: Audits orbit assumptions.
category: analysis
---

# Orbit Audit
""",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    tools = build_basic_tools(workspace, skill_registry=registry)
    discover_tool = next(tool for tool in tools if tool.name == "discover_skill_manifests")

    result = discover_tool.invoke({"roots": [str(skill_root)]})

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["manifests"][0]["name"] == "orbit-audit"
    assert result["manifests"][0]["executable"] is False


def test_basic_tools_install_skill_from_path():
    workspace = _workspace("skill_install")
    source_dir = workspace / "source" / "sensor-check"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "SKILL.md").write_text(
        """---
name: sensor-check
description: Checks sensor simulation assumptions.
category: analysis
---

# Sensor Check
""",
        encoding="utf-8",
    )
    install_root = workspace / "installed"
    registry = SkillRegistry()
    tools = build_basic_tools(
        workspace,
        skill_registry=registry,
        skill_install_dir=install_root,
    )
    install_tool = next(tool for tool in tools if tool.name == "install_skill_from_path")
    use_tool = next(tool for tool in tools if tool.name == "use_skill")

    result = install_tool.invoke({"path": str(source_dir), "overwrite": True})
    use_result = use_tool.invoke({"name": "sensor-check"})

    assert result["status"] == "ok"
    assert result["manifest"]["name"] == "sensor-check"
    assert Path(result["installed_path"], "SKILL.md").exists()
    assert use_result["status"] == "error"
    assert use_result["result"]["error_code"] == "SKILL_NOT_EXECUTABLE"


def test_cli_engine_uses_langchain_even_if_legacy_mode_is_requested():
    class Agent:
        def __init__(self):
            self.calls = []

        def run_langchain(self, task, stream_callback=None):
            self.calls.append((task, stream_callback is not None))
            return "langchain result"

    agent = Agent()
    engine = CEOEngine(
        llm=None,
        agent=agent,
        console=None,
        stats=Stats(),
        mode="fast",
        stream=False,
    )

    result = engine.execute("task")

    assert result == "langchain result"
    assert agent.calls == [("task", False)]


def test_aerospace_agent_langchain_memory_persists_across_calls():
    llm = OneShotLLM("persistent answer")
    agent = AerospaceAgent(llm=llm, tools=[])

    agent.run_langchain("first question")
    agent.run_langchain("second question")

    contents = [message["content"] for message in llm.messages]
    assert "first question" in contents
    assert "persistent answer" in contents
    assert contents[-1] == "second question"
