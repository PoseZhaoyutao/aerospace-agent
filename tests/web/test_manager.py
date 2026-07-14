import threading

import pytest

from aerospace_agent.langgraph_agent.schema import AgentOutput, RunStatus
from aerospace_agent.web.manager import AgentRuntimeManager, ThreadScopeError


class FakeAgent:
    def __init__(self, output=None):
        self.output = output or AgentOutput(status=RunStatus.SUCCESS, answer="ok")
        self.calls = []
        self.close_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, message, thread_id=None, context=None):
        self.calls.append((message, thread_id))
        self.started.set()
        if context and context.get("block"):
            self.release.wait(timeout=2)
        return self.output

    def close(self):
        self.close_calls += 1


def test_manager_creates_server_owned_threads_and_maps_output():
    agent = FakeAgent()
    manager = AgentRuntimeManager(
        agent_factory=lambda: agent,
        project_id="project-1",
        workspace_id="workspace-1",
    )
    manager.start()
    summary = manager.create_thread(title="Test")

    event = manager.run(summary.thread_id, request_id="req-1", message="hello")

    assert summary.project_id == "project-1"
    assert summary.title == "Test"
    assert event.type == "run.completed"
    assert event.status == "success"
    assert agent.calls == [("hello", summary.thread_id)]

    with pytest.raises(ThreadScopeError):
        manager.run("not-owned", request_id="req-2", message="no")


def test_manager_rejects_same_thread_concurrency():
    agent = FakeAgent()
    manager = AgentRuntimeManager(
        agent_factory=lambda: agent,
        project_id="project-1",
        workspace_id="workspace-1",
    )
    manager.start()
    thread_id = manager.create_thread().thread_id
    holder = {}

    def run_first():
        holder["event"] = manager.run(
            thread_id, request_id="req-1", message="hello", context={"block": True}
        )

    first = threading.Thread(target=run_first)
    first.start()
    assert agent.started.wait(timeout=1)
    with pytest.raises(RuntimeError, match="already running"):
        manager.run(thread_id, request_id="req-2", message="duplicate")
    agent.release.set()
    first.join(timeout=2)
    assert holder["event"].status == "success"
    assert len(agent.calls) == 1


def test_manager_shutdown_is_exactly_once():
    agent = FakeAgent()
    manager = AgentRuntimeManager(
        agent_factory=lambda: agent,
        project_id="project-1",
        workspace_id="workspace-1",
    )
    manager.start()
    manager.shutdown()
    manager.shutdown()
    assert agent.close_calls == 1
