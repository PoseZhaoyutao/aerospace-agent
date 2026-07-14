from __future__ import annotations

import inspect
from pathlib import Path

from aerospace_agent.langgraph_agent.agent_core.execution import ExecutionRegistry
from aerospace_agent.langgraph_agent.agent_core.git_service import GitService
from aerospace_agent.langgraph_agent.agent_core.journal import OperationJournal
from aerospace_agent.langgraph_agent.agent_core.tools.files import FileService
from aerospace_agent.langgraph_agent.agent_core.tools.system import (
    CORE_TOOL_NAMES,
    CoreToolServices,
    build_core_tool_catalog,
)
from aerospace_agent.langgraph_agent.agent_core.tools.terminal import TerminalService
from aerospace_agent.langgraph_agent.agent_core.tools.web import WebService


EXPECTED_GROUPS = {
    "file": 11,
    "terminal": 3,
    "browser": 4,
    "web": 3,
    "schedule": 4,
    "memory": 6,
    "git": 7,
    "workflow": 2,
    "capability": 3,
}


def test_inventory_is_stable_self_contained_and_has_at_least_thirty_tools(tmp_path: Path):
    empty = build_core_tool_catalog(tmp_path, CoreToolServices())
    with_services = build_core_tool_catalog(
        tmp_path,
        CoreToolServices(
            files=FileService(tmp_path),
            terminal=TerminalService(tmp_path, allowed_commands=["python"]),
            git=GitService(tmp_path),
        ),
    )

    assert tuple(item.tool_name for item in empty.entries()) == CORE_TOOL_NAMES
    assert tuple(item.tool_name for item in with_services.entries()) == CORE_TOOL_NAMES
    assert len(CORE_TOOL_NAMES) >= 30
    assert len(CORE_TOOL_NAMES) == len(set(CORE_TOOL_NAMES))
    for prefix, count in EXPECTED_GROUPS.items():
        assert sum(name.startswith(prefix + ".") for name in CORE_TOOL_NAMES) == count


def test_available_status_has_exact_executor_parity_and_current_repo_source(tmp_path: Path):
    catalog = build_core_tool_catalog(
        tmp_path,
        CoreToolServices(
            files=FileService(tmp_path),
            terminal=TerminalService(tmp_path, allowed_commands=["python"]),
            git=GitService(tmp_path),
        ),
    )

    available = {item.tool_name for item in catalog.entries() if item.manifest.status == "available"}
    assert available == set(catalog.executable_names())
    assert {item.tool_name for item in catalog.entries() if item.manifest.status != "available"}.isdisjoint(
        catalog.executable_names()
    )
    assert {
        "file.read", "file.read_lines", "file.write", "file.append", "file.list",
        "file.search", "file.info", "file.mkdir", "file.copy", "file.move", "file.delete",
    }.issubset(available)
    # The workspace fixture is not a valid Git repository, so Git must remain truthful.
    assert not any(name.startswith("git.") for name in available)
    for binding in catalog._validated_bindings_for_bootstrap():
        source = Path(inspect.getsourcefile(binding.handler) or "").resolve()
        source.relative_to(Path.cwd().resolve())
        assert binding.entrypoint.startswith(
            (
                "aerospace_agent.mcp.tools.core_tool_adapters.",
                "aerospace_agent.mcp.tools.space_tools.",
            )
        )


def test_catalog_registers_only_available_handlers_in_execution_registry(tmp_path: Path):
    workspace = Path.cwd().resolve()
    catalog = build_core_tool_catalog(
        workspace,
        CoreToolServices(
            files=FileService(
                workspace,
                journal=OperationJournal(
                    tmp_path / "journal.sqlite3",
                    backup_dir=tmp_path / "preimages",
                ),
            )
        ),
    )
    registry = ExecutionRegistry(
        workspace,
        audit_database_path=tmp_path / ".agent_core" / "audit.sqlite3",
    )

    registered = catalog.register_into(registry)

    assert set(registered) == set(catalog.executable_names())
    for tool_name in registered:
        entry = catalog.get(tool_name)
        snapshot = registry.snapshot(entry.manifest.capability_id)
        assert snapshot.capability_id == entry.manifest.capability_id
    assert "file.write" not in registered


def test_every_entry_exports_strict_json_input_and_output_schemas(tmp_path: Path):
    catalog = build_core_tool_catalog(tmp_path, CoreToolServices())

    definitions = catalog.definitions()

    assert [item["name"] for item in definitions] == list(CORE_TOOL_NAMES)
    for definition in definitions:
        assert definition["inputSchema"]["type"] == "object"
        assert definition["inputSchema"].get("additionalProperties") is False
        assert definition["outputSchema"]["type"] == "object"
        assert definition["status"] in {"available", "unavailable"}


def test_configured_public_search_provider_is_executable(tmp_path: Path):
    catalog = build_core_tool_catalog(
        tmp_path,
        CoreToolServices(
            web=WebService(
                tmp_path,
                search_providers=[
                    {"name": "public", "endpoint": "https://search.example.test"}
                ],
            )
        ),
    )

    entry = catalog.get("web.search")

    assert entry.manifest.status == "available"
    assert "web.search" in catalog.executable_names()

