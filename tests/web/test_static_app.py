from pathlib import Path

from fastapi.testclient import TestClient

from aerospace_agent.web.app import create_app
from aerospace_agent.web.manager import AgentRuntimeManager


class FakeAgent:
    model_name = "test-model"

    def close(self):
        pass


def make_manager():
    return AgentRuntimeManager(
        agent_factory=FakeAgent,
        project_id="project-1",
        workspace_id="workspace-1",
    )


def test_static_root_serves_built_index(tmp_path: Path):
    (tmp_path / "index.html").write_text("<html>webui</html>", encoding="utf-8")
    with TestClient(create_app(manager=make_manager(), static_dir=tmp_path)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.text == "<html>webui</html>"


def test_missing_static_build_returns_structured_503(tmp_path: Path):
    with TestClient(create_app(manager=make_manager(), static_dir=tmp_path)) as client:
        response = client.get("/")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WEBUI_STATIC_NOT_BUILT"


def test_disabled_webui_returns_structured_503(tmp_path: Path):
    with TestClient(
        create_app(manager=make_manager(), static_dir=tmp_path, webui_enabled=False)
    ) as client:
        response = client.get("/")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WEBUI_DISABLED"


def test_allow_lan_and_non_loopback_are_rejected_without_authentication(tmp_path: Path):
    (tmp_path / "index.html").write_text("<html>webui</html>", encoding="utf-8")
    for kwargs in ({"allow_lan": True}, {"host": "0.0.0.0"}):
        with TestClient(create_app(manager=make_manager(), static_dir=tmp_path, **kwargs)) as client:
            response = client.get("/")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "WEBUI_LAN_NOT_ALLOWED"
