from __future__ import annotations

import pytest

from aerospace_agent.langgraph_agent.agent_core.capabilities import CapabilityRegistry
from aerospace_agent.langgraph_agent.agent_core.models import CapabilityManifest


def _manifest(
    capability_id: str,
    *,
    status: str = "available",
    intents: list[str] | None = None,
    tool_names: list[str] | None = None,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_id=capability_id,
        version="1.0.0",
        category="basic" if status == "available" else "domain",
        status=status,
        intents=intents or ["files"],
        tool_names=tool_names or [],
        risk_level="read_only",
        source="aerospace_agent.mcp.tools",
    )


def test_registry_rejects_duplicate_capability_ids() -> None:
    manifest = _manifest("files")

    with pytest.raises(ValueError, match="duplicate capability_id"):
        CapabilityRegistry([manifest, manifest])


def test_discovery_includes_interface_only_but_execution_candidates_do_not() -> None:
    available = _manifest("files", intents=["files"])
    interface = _manifest("simulation", status="interface_only", intents=["simulation"])
    registry = CapabilityRegistry([available, interface])

    assert [item.capability_id for item in registry.list_manifests()] == ["files", "simulation"]
    assert [item.capability_id for item in registry.candidates_for_intents(["simulation"])] == []
    assert registry.discover("simulation").status == "interface_only"


def test_registry_never_returns_a_callable_and_returns_defensive_copies() -> None:
    original = _manifest("files", tool_names=["file.read"])
    registry = CapabilityRegistry([original])

    fetched = registry.get("files")
    assert not callable(fetched)
    fetched.tool_names.append("file.delete")

    assert registry.get("files").tool_names == ["file.read"]


def test_registry_rejects_sources_outside_current_repository_import_roots() -> None:
    manifest = _manifest("external")
    manifest.source = "neighbor_project.research_tools"

    with pytest.raises(ValueError, match="allowed current-repository roots"):
        CapabilityRegistry([manifest])


def test_candidates_are_relevant_available_and_hard_limited_to_twelve() -> None:
    manifests = [
        _manifest(f"files-{index:02d}", intents=["files", f"intent-{index:02d}"])
        for index in range(15)
    ]
    manifests.append(_manifest("disabled", status="disabled", intents=["files"]))
    registry = CapabilityRegistry(manifests)

    candidates = registry.candidates_for_intents(["files"], limit=12)

    assert len(candidates) == 12
    assert all(item.status == "available" for item in candidates)
    assert all("files" in item.intents for item in candidates)


def test_candidate_limit_cannot_exceed_planner_limit() -> None:
    registry = CapabilityRegistry([_manifest("files")])

    with pytest.raises(ValueError, match="between 1 and 12"):
        registry.candidates_for_intents(["files"], limit=13)


def test_planner_candidate_budget_counts_tools_not_only_manifests() -> None:
    registry = CapabilityRegistry(
        [_manifest("files", intents=["files"], tool_names=[f"file.tool_{i}" for i in range(20)])]
    )

    candidates = registry.planner_candidates_for_intents(["files"])

    assert sum(len(candidate.executor_names) for candidate in candidates) == 12


def test_repository_loader_rejects_manifest_path_escape(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """
capability_id: files
version: 1.0.0
category: basic
status: available
intents: [files]
tool_names: [file.read]
risk_level: read_only
required_dependencies: []
validators: []
source: aerospace_agent.mcp.tools
""".strip(),
        encoding="utf-8",
    )
    registry = CapabilityRegistry.from_repository(tmp_path, [manifest_path])
    assert registry.get("files").source == "aerospace_agent.mcp.tools"

    with pytest.raises(ValueError, match="outside workspace"):
        CapabilityRegistry.from_repository(tmp_path / "nested", [manifest_path])


def test_available_workflow_requires_external_approval_verifier() -> None:
    manifest = _manifest("workflow", intents=["workflow"])
    manifest.category = "workflow"

    with pytest.raises(ValueError, match="approval verification"):
        CapabilityRegistry([manifest])

    registry = CapabilityRegistry([manifest], approval_verifier=lambda item: item.capability_id == "workflow")
    assert registry.get("workflow").status == "available"


def test_request_matching_keeps_relevant_thirteenth_executor_in_same_manifest() -> None:
    tool_names = [f"tool.unrelated_{index}" for index in range(12)] + ["tool.target"]
    registry = CapabilityRegistry(
        [_manifest("many-tools", intents=["tools"], tool_names=tool_names)]
    )

    candidates = registry.planner_candidates_for_request("请调用 tool.target")

    assert candidates[0].executor_names == ["tool.target"]


@pytest.mark.parametrize("first_status", ["available", "interface_only"])
def test_duplicate_executor_ownership_is_rejected_independent_of_registration_order(
    first_status: str,
) -> None:
    second_status = "interface_only" if first_status == "available" else "available"
    first = _manifest("first", status=first_status, tool_names=["file.read"])
    second = _manifest("second", status=second_status, tool_names=["file.read"])

    with pytest.raises(ValueError, match="duplicate executor name"):
        CapabilityRegistry([first, second])

