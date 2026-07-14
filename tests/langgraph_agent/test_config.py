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
    assert settings.workspace_root == tmp_path
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


def test_runtime_services_factory_composes_runtime_handles_without_closing_them(tmp_path):
    from aerospace_agent.langgraph_agent.graph import ServiceBundle
    from aerospace_agent.langgraph_agent.services.context import ContextService
    from aerospace_agent.langgraph_agent.services.evolution import EvolutionService
    from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    class Gateway:
        closed = False

        def __init__(self):
            self.close_calls = 0

        def list_tools(self):
            return []

        def call_tool(self, request):
            raise AssertionError(f"unexpected tool call: {request}")

        def close(self):
            self.close_calls += 1
            self.closed = True

    settings = load_settings(workspace=tmp_path)
    gateway = Gateway()
    built_with = []

    def build_gateway(mcp_settings):
        built_with.append(mcp_settings)
        return gateway, ["explicit degraded-mode warning"]

    runtime = RuntimeServicesFactory(
        settings,
        project_root=tmp_path,
        gateway_builder=build_gateway,
    ).create()

    assert isinstance(runtime.bundle, ServiceBundle)
    assert isinstance(runtime.context, ContextService)
    assert isinstance(runtime.knowledge, KnowledgeService)
    assert isinstance(runtime.evolution, EvolutionService)
    assert runtime.bundle.context is runtime.context
    assert runtime.bundle.knowledge is runtime.knowledge
    assert runtime.bundle.evolution is runtime.evolution
    assert runtime.bundle.mcp_gateway is gateway
    assert runtime.bundle.llm is runtime.llm
    assert runtime.bundle.planner is not None
    assert runtime.bundle.planner.llm is runtime.llm
    assert runtime.llm.endpoint == settings.llm.endpoint
    assert runtime.llm.model == settings.llm.model
    assert runtime.warnings == ("explicit degraded-mode warning",)
    assert built_with == [settings.mcp]
    assert gateway.close_calls == 0


def test_runtime_factory_materializes_only_declared_workspace_directories(tmp_path):
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    class Gateway:
        closed = False

        def list_tools(self):
            return []

        def call_tool(self, request):
            raise AssertionError(f"unexpected tool call: {request}")

        def close(self):
            self.closed = True

    settings = load_settings(workspace=tmp_path)
    RuntimeServicesFactory(
        settings,
        project_root=tmp_path,
        gateway_builder=lambda _settings: (Gateway(), []),
    ).create()

    expected = (
        "knowledge",
        "memory",
        "evolved_skills",
        "workflows/evolved",
        "data/langgraph/rag",
        "data/langgraph/artifacts",
        "data/langgraph/evolution",
    )
    assert all((tmp_path / relative).is_dir() for relative in expected)


def test_runtime_services_factory_uses_explicit_root_for_custom_knowledge_path(tmp_path):
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    class Gateway:
        closed = False

        def list_tools(self):
            return []

        def call_tool(self, request):
            raise AssertionError(f"unexpected tool call: {request}")

        def close(self):
            self.closed = True

    settings = AgentSettings.from_mapping(
        {"knowledge": {"workspace": "custom/wiki"}},
        workspace=tmp_path,
    )
    runtime = RuntimeServicesFactory(
        settings,
        project_root=tmp_path,
        gateway_builder=lambda _settings: (Gateway(), []),
    ).create()

    assert runtime.knowledge.wiki_root == tmp_path / "custom" / "wiki"
    assert runtime.context.workspace == tmp_path
    assert runtime.context.artifact_dir == tmp_path / "data" / "langgraph" / "artifacts"
    assert runtime.evolution.workspace == tmp_path


def test_runtime_services_factory_rejects_settings_from_a_different_project_root(tmp_path):
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    settings = load_settings(workspace=tmp_path)

    with pytest.raises(ValueError, match="project root"):
        RuntimeServicesFactory(settings, project_root=tmp_path / "other-project")


def test_runtime_services_factory_rejects_an_ancestor_of_the_selected_root(tmp_path):
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    selected_root = tmp_path / "selected-project"
    settings = load_settings(workspace=selected_root)

    with pytest.raises(ValueError, match="selected workspace"):
        RuntimeServicesFactory(settings, project_root=tmp_path)


def test_retrieval_confidence_threshold_defaults_to_point_six(tmp_path):
    settings = load_settings(workspace=tmp_path)

    assert settings.knowledge.retrieval_confidence_threshold == 0.60


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_retrieval_confidence_threshold_is_probability(tmp_path, value):
    with pytest.raises(ValueError):
        AgentSettings.from_mapping(
            {"knowledge": {"retrieval_confidence_threshold": value}},
            workspace=tmp_path,
        )


def test_webui_settings_have_local_defaults_and_are_strict(tmp_path):
    settings = AgentSettings.from_mapping({}, workspace=tmp_path)

    assert settings.webui.enabled is True
    assert settings.webui.host == "127.0.0.1"
    assert settings.webui.port == 8765
    assert settings.webui.allow_lan is False
    assert settings.webui.auth_token_env is None

    with pytest.raises(ValueError):
        AgentSettings.from_mapping({"webui": {"unknown": True}}, workspace=tmp_path)

    with pytest.raises(ValueError):
        AgentSettings.from_mapping({"webui": {"port": 0}}, workspace=tmp_path)

    with pytest.raises(ValueError, match="authentication is not implemented"):
        AgentSettings.from_mapping(
            {"webui": {"auth_token_env": "WEBUI_TOKEN"}}, workspace=tmp_path
        )


def test_webui_allow_lan_is_parseable_but_preserved_for_application_rejection(tmp_path):
    settings = AgentSettings.from_mapping(
        {"webui": {"allow_lan": True}}, workspace=tmp_path
    )
    assert settings.webui.allow_lan is True
    assert settings.webui.host == "127.0.0.1"


def test_webui_yaml_defaults_match_default_mapping(tmp_path):
    import yaml

    raw = yaml.safe_load(Path("config/langgraph_agent.yaml").read_text(encoding="utf-8"))
    from aerospace_agent.langgraph_agent.config import _default_mapping

    assert raw["webui"] == _default_mapping()["webui"]


def test_web_provider_chain_and_browser_settings_are_strict_and_workspace_independent(tmp_path):
    settings = AgentSettings.from_mapping(
        {
            "web": {
                "search_providers": [
                    {
                        "name": "primary",
                        "endpoint": "https://search.example.test/v1",
                        "api_key_env": "SEARCH_API_KEY",
                    }
                ],
                "default_search_provider": "primary",
            },
            "browser": {"playwright_enabled": True},
        },
        workspace=tmp_path,
    )

    assert settings.web.default_search_provider == "primary"
    assert settings.web.search_providers[0].api_key_env == "SEARCH_API_KEY"
    assert settings.browser.playwright_enabled is True


def test_unknown_web_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        AgentSettings.from_mapping({"web": {"unknown": True}}, workspace=tmp_path)


def test_llm_provider_settings_allow_multiple_external_api_kinds(tmp_path):
    settings = AgentSettings.from_mapping(
        {
            "llm": {
                "provider": "anthropic",
                "providers": [
                    {
                        "name": "anthropic",
                        "kind": "anthropic",
                        "endpoint": "https://api.anthropic.test",
                        "model": "claude-test",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                    {
                        "name": "deepseek",
                        "kind": "openai_compatible",
                        "endpoint": "https://api.deepseek.test/v1",
                        "model": "deepseek-chat",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            }
        },
        workspace=tmp_path,
    )

    assert [item.kind for item in settings.llm.providers] == ["anthropic", "openai_compatible"]
