from fastapi.testclient import TestClient

from aerospace_agent.langgraph_agent.schema import AgentOutput, RunStatus
from aerospace_agent.web.app import create_app
from aerospace_agent.web.manager import AgentRuntimeManager


class FakeAgent:
    def run(self, message, thread_id=None, context=None):
        return AgentOutput(status=RunStatus.SUCCESS, answer=f"echo:{message}")

    def close(self):
        pass


def make_client():
    manager = AgentRuntimeManager(
        agent_factory=FakeAgent,
        project_id="project-1",
        workspace_id="workspace-1",
    )
    return TestClient(create_app(manager=manager))


def test_health_and_thread_routes_return_strict_versioned_shapes():
    with make_client() as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["schema_version"] == "1.0.0"

        created = client.post("/api/v1/threads", json={"title": "Mission"})
        assert created.status_code == 201
        thread = created.json()
        assert thread["title"] == "Mission"
        assert thread["project_id"] == "project-1"

        listed = client.get("/api/v1/threads")
        assert listed.status_code == 200
        assert listed.json()["threads"][0]["thread_id"] == thread["thread_id"]


def test_thread_scope_and_request_injection_are_rejected():
    with make_client() as client:
        response = client.post(
            "/api/v1/threads",
            json={"title": "x", "project_id": "other", "workspace": "C:\\secret"},
        )
        assert response.status_code == 422

        response = client.get("/api/v1/threads/not-owned/history")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "THREAD_NOT_FOUND"
