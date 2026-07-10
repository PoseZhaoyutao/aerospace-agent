import ast
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.config import AgentSettings, load_settings


REQUIRED = {
    "langgraph": ("1.0", "2.0"),
    "langgraph-checkpoint-sqlite": ("3.0", "4.0"),
    "langchain-core": ("1.0", "2.0"),
    "pydantic": ("2.0", "3.0"),
    "mcp": ("1.0", "2.0"),
}


def test_runtime_dependencies_are_bounded_in_requirements_and_setup():
    active_requirement_lines = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    requirements = {
        line.split(">=")[0]: line
        for line in active_requirement_lines
        if ">=" in line
    }
    tree = ast.parse(Path("setup.py").read_text(encoding="utf-8"))
    setup_specs = {
        item.value.split(">=")[0]: item.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "install_requires"
        for item in node.value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    for name, (low, high) in REQUIRED.items():
        exact = f"{name}>={low},<{high}"
        assert sum(line.split(">=")[0] == name for line in active_requirement_lines) == 1
        assert requirements[name] == exact
        assert setup_specs[name] == exact


def test_settings_resolve_paths_inside_workspace(tmp_path):
    settings = load_settings(workspace=tmp_path)
    assert settings.knowledge.workspace == tmp_path / "knowledge"
    assert settings.checkpoint.path == tmp_path / "data/langgraph/checkpoints.sqlite"
    assert settings.knowledge.data_dir == tmp_path / "data/langgraph/rag"
    assert settings.context.artifacts_dir == tmp_path / "data/langgraph/artifacts"
    assert settings.evolution.data_dir == tmp_path / "data/langgraph/evolution"
    assert settings.knowledge.graph_output == tmp_path / "reports/knowledge_graph.html"
    assert settings.evolution.allowed_roots == [
        tmp_path / "knowledge",
        tmp_path / "memory",
        tmp_path / "evolved_skills",
        tmp_path / "workflows/evolved",
    ]


def test_settings_reject_path_escape(tmp_path):
    escaped = [
        {"knowledge": {"workspace": "../outside"}},
        {"knowledge": {"data_dir": "../outside"}},
        {"checkpoint": {"path": "../outside/checkpoints.sqlite"}},
        {"context": {"artifacts_dir": "../outside"}},
        {"evolution": {"data_dir": "../outside"}},
        {"evolution": {"allowed_roots": ["../outside"]}},
    ]
    for mapping in escaped:
        with pytest.raises(ValueError, match="workspace"):
            AgentSettings.from_mapping(mapping, workspace=tmp_path)


def test_explicit_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AEROSPACE_LANGGRAPH_CONFIG", str(tmp_path / "custom.yaml"))
    monkeypatch.setenv("AEROSPACE_LOCAL_LLM_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("AEROSPACE_LOCAL_LLM_MODEL", "test-model")
    (tmp_path / "custom.yaml").write_text("llm: {}\n", encoding="utf-8")
    settings = load_settings(workspace=tmp_path)
    assert settings.llm.endpoint == "http://127.0.0.1:9000/v1"
    assert settings.llm.model == "test-model"


def test_missing_explicit_environment_config_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("AEROSPACE_LANGGRAPH_CONFIG", "missing.yaml")
    with pytest.raises(FileNotFoundError, match="configuration not found"):
        load_settings(workspace=tmp_path)


def test_relative_environment_config_is_anchored_to_workspace(monkeypatch, tmp_path):
    config_path = tmp_path / "configs" / "custom.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "llm:\n  endpoint: http://127.0.0.1:9100/v1\n  model: relative-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AEROSPACE_LANGGRAPH_CONFIG", "configs/custom.yaml")
    settings = load_settings(workspace=tmp_path)
    assert settings.llm.endpoint == "http://127.0.0.1:9100/v1"
    assert settings.llm.model == "relative-model"
