from fastapi.testclient import TestClient

from aerospace_agent.langgraph_agent.schema import AgentOutput, RunStatus
from aerospace_agent.web.app import create_app
from aerospace_agent.web.manager import AgentRuntimeManager


class FakeAgent:
    def run(self, message, thread_id=None, context=None):
        return AgentOutput(status=RunStatus.PARTIAL, answer=f"partial:{message}")

    def close(self):
        pass


def test_websocket_emits_ready_started_and_truthful_terminal_event():
    manager = AgentRuntimeManager(
        agent_factory=FakeAgent,
        project_id="project-1",
        workspace_id="workspace-1",
    )
    manager.start()
    thread_id = manager.create_thread().thread_id
    with TestClient(create_app(manager=manager)) as client:
        with client.websocket_connect("/api/v1/ws") as socket:
            assert socket.receive_json()["type"] == "connection.ready"
            socket.send_json(
                {
                    "schema_version": "1.0.0",
                    "type": "run.start",
                    "request_id": "req-1",
                    "thread_id": thread_id,
                    "message": "hello",
                }
            )
            assert socket.receive_json()["type"] == "run.accepted"
            assert socket.receive_json()["type"] == "run.started"
            terminal = socket.receive_json()
            assert terminal["type"] == "run.completed"
            assert terminal["status"] == "partial"
            assert terminal["answer"] == "partial:hello"


def test_websocket_rejects_unknown_thread_and_malformed_payload():
    manager = AgentRuntimeManager(
        agent_factory=FakeAgent,
        project_id="project-1",
        workspace_id="workspace-1",
    )
    with TestClient(create_app(manager=manager)) as client:
        with client.websocket_connect("/api/v1/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "run.start", "message": "missing fields"})
            error = socket.receive_json()
            assert error["type"] == "run.failed"
            assert error["status"] == "error"
