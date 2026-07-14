"""Stable, self-contained Agent Core tool catalog.

Inventory and executability are deliberately separate.  Every name is always
discoverable, while ``available`` means a concrete current-repository adapter
can be registered in :class:`ExecutionRegistry` at this instant.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, create_model

from aerospace_agent.mcp.tools.core_tool_adapters import CoreToolAdapters
from aerospace_agent.mcp.tools.space_tools import SPACE_TOOL_CALLS, SPACE_TOOL_SPECS

from ..execution import ExecutionRegistry
from ..models import CapabilityManifest, FrozenContractModel


BASIC_TOOL_NAMES = (
    "file.read", "file.read_lines", "file.write", "file.append", "file.list",
    "file.search", "file.info", "file.mkdir", "file.copy", "file.move", "file.delete",
    "terminal.run", "terminal.status", "terminal.cancel",
    "browser.open", "browser.follow_link", "browser.extract", "browser.screenshot",
    "web.search", "web.fetch", "web.download",
    "schedule.create", "schedule.list", "schedule.cancel", "schedule.run_due",
    "memory.remember", "memory.search", "memory.list", "memory.update", "memory.forget",
    "memory.clear",
    "git.status", "git.diff", "git.log", "git.branch_info", "git.create_checkpoint",
    "git.revert_commit", "git.restore_paths",
    "workflow.list", "workflow.run",
    "capability.list", "capability.describe", "capability.acquire",
)
SPACE_BASIC_TOOL_NAMES = tuple(item["name"] for item in SPACE_TOOL_SPECS)
CORE_TOOL_NAMES = BASIC_TOOL_NAMES + SPACE_BASIC_TOOL_NAMES


@dataclass(frozen=True)
class CoreToolServices:
    files: Any | None = None
    terminal: Any | None = None
    browser: Any | None = None
    web: Any | None = None
    scheduler: Any | None = None
    memory: Any | None = None
    git: Any | None = None
    workflows: Any | None = None
    capabilities: Any | None = None


class ToolCatalogEntry(FrozenContractModel):
    tool_name: str
    manifest: CapabilityManifest
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    availability_reason: str
    requires_confirmation: bool = False
    recovery_class: Literal["read_only", "reversible", "compensatable", "manual_recovery"]
    path_fields: tuple[str, ...] = ()
    adapter_sha256: str | None = None


@dataclass(frozen=True)
class _Binding:
    tool_name: str
    handler: Callable[..., Any]
    input_model: type[BaseModel]
    entrypoint: str
    adapter_path: Path
    recovery_class: str
    path_fields: tuple[str, ...]
    requires_confirmation: bool


_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
}


def _p(type_name: str, *, default: Any = ..., description: str = "") -> tuple[Any, Any]:
    annotation = _TYPE_MAP[type_name]
    if default is None:
        annotation = annotation | None
    return annotation, Field(default=default, description=description)


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyInput(_StrictInput):
    pass


class FileReadInput(_StrictInput):
    path: str
    max_bytes: int = 1_000_000


class FileReadLinesInput(_StrictInput):
    path: str
    start_line: int = 1
    end_line: int | None = None
    max_bytes: int = 1_000_000


class FileListInput(_StrictInput):
    path: str = "."
    max_results: int = 1_000


class FileSearchInput(_StrictInput):
    path: str
    query: str
    max_results: int = 100


class FileInfoInput(_StrictInput):
    path: str


class TerminalRunInput(_StrictInput):
    argv: list[Any]
    cwd: str | None = None
    env: dict[str, Any] | None = None
    timeout_s: float = 120.0
    max_output_chars: int = 100_000
    background: bool = False


class ProcessInput(_StrictInput):
    process_id: str


class BrowserOpenInput(_StrictInput):
    url: str


class BrowserFollowInput(_StrictInput):
    page_id: str
    link_id: int


class BrowserExtractInput(_StrictInput):
    page_id: str
    max_chars: int = 100_000


class BrowserScreenshotInput(_StrictInput):
    page_id: str
    target_path: str


class WebSearchInput(_StrictInput):
    query: str
    max_results: int = 10


class WebFetchInput(_StrictInput):
    url: str
    max_bytes: int = 1_000_000
    hard_limit_bytes: int = 5_000_000


class WebDownloadInput(_StrictInput):
    url: str
    target_path: str
    expected_sha256: str | None = None
    overwrite: bool = False


class ScheduleCreateInput(_StrictInput):
    kind: str
    due_at: str
    message: str | None = None
    workflow_id: str | None = None
    workflow_version: str | None = None
    inputs: dict[str, Any] | None = None
    max_retries: int = 0
    retry_delay_seconds: int = 30


class ScheduleCancelInput(_StrictInput):
    job_id: str
    expected_version: int


class MemoryRememberInput(_StrictInput):
    kind: str
    content: str
    source_checkpoints: list[Any]
    source_content_hash: str
    truth_status: str
    confidence: float


class MemorySearchInput(_StrictInput):
    query: str
    include_history: bool = False
    limit: int = 20


class MemoryListInput(_StrictInput):
    include_history: bool = False
    limit: int = 100


class MemoryUpdateInput(_StrictInput):
    memory_id: str
    content: str
    source_checkpoints: list[Any]
    source_content_hash: str
    truth_status: str
    confidence: float


class MemoryIdInput(_StrictInput):
    memory_id: str


class GitPathsInput(_StrictInput):
    paths: list[Any] | None = None


class GitDiffInput(_StrictInput):
    paths: list[Any] | None = None
    staged: bool = False
    revision: str | None = None


class GitLogInput(_StrictInput):
    max_count: int = 20
    paths: list[Any] | None = None


class GitCheckpointInput(_StrictInput):
    message: str
    paths: list[Any]


class GitRevertInput(_StrictInput):
    commit: str
    paths: list[Any]


class GitRestoreInput(_StrictInput):
    paths: list[Any]
    source: str = "HEAD"


class CapabilityDescribeInput(_StrictInput):
    capability_id: str


class SpaceEnvironmentInput(_StrictInput):
    engines: list[str] | None = None
    timeout_seconds: float = 1.0


class SpaceTimeInput(_StrictInput):
    value: str | float
    from_scale: str
    from_format: str
    to_scale: str
    to_format: str


class SpaceFrameInput(_StrictInput):
    state_dict: dict[str, Any]
    target_frame: str
    target_center: str | None = None


class SpaceOrbitRepresentationInput(_StrictInput):
    state_dict: dict[str, Any]
    target_representation: str
    mu: float = 3.986004418e14


class SpacePropagationInput(_StrictInput):
    initial_state_dict: dict[str, Any]
    force_model_dict: dict[str, Any]
    duration_s: float
    output_step_s: float | None = None
    engine: str = "auto"


class SpaceCrossValidationInput(_StrictInput):
    task_spec: dict[str, Any]
    engines: list[str] | None = None
    existing_results: dict[str, Any] | None = None
    thresholds: dict[str, Any] | None = None


def _contract_fields() -> dict[str, dict[str, tuple[Any, Any]]]:
    empty: dict[str, tuple[Any, Any]] = {}
    return {
        "file.read": {"path": _p("string"), "max_bytes": _p("integer", default=1_000_000)},
        "file.read_lines": {"path": _p("string"), "start_line": _p("integer", default=1), "end_line": _p("integer", default=None), "max_bytes": _p("integer", default=1_000_000)},
        "file.write": {"path": _p("string"), "content": _p("string"), "overwrite": _p("boolean", default=False)},
        "file.append": {"path": _p("string"), "content": _p("string")},
        "file.list": {"path": _p("string", default="."), "max_results": _p("integer", default=1_000)},
        "file.search": {"path": _p("string"), "query": _p("string"), "max_results": _p("integer", default=100)},
        "file.info": {"path": _p("string")},
        "file.mkdir": {"path": _p("string")},
        "file.copy": {"source": _p("string"), "destination": _p("string"), "overwrite": _p("boolean", default=False)},
        "file.move": {"source": _p("string"), "destination": _p("string"), "overwrite": _p("boolean", default=False)},
        "file.delete": {"path": _p("string"), "recursive": _p("boolean", default=False)},
        "terminal.run": {"argv": _p("array"), "cwd": _p("string", default=None), "env": _p("object", default=None), "timeout_s": _p("number", default=120.0), "max_output_chars": _p("integer", default=100_000), "background": _p("boolean", default=False)},
        "terminal.status": {"process_id": _p("string")},
        "terminal.cancel": {"process_id": _p("string")},
        "browser.open": {"url": _p("string")},
        "browser.follow_link": {"page_id": _p("string"), "link_id": _p("integer")},
        "browser.extract": {"page_id": _p("string"), "max_chars": _p("integer", default=100_000)},
        "browser.screenshot": {"page_id": _p("string"), "target_path": _p("string")},
        "web.search": {"query": _p("string"), "max_results": _p("integer", default=10)},
        "web.fetch": {"url": _p("string"), "max_bytes": _p("integer", default=1_000_000), "hard_limit_bytes": _p("integer", default=5_000_000)},
        "web.download": {"url": _p("string"), "target_path": _p("string"), "expected_sha256": _p("string", default=None), "overwrite": _p("boolean", default=False)},
        "schedule.create": {"kind": _p("string"), "due_at": _p("string"), "message": _p("string", default=None), "workflow_id": _p("string", default=None), "workflow_version": _p("string", default=None), "inputs": _p("object", default=None), "max_retries": _p("integer", default=0), "retry_delay_seconds": _p("integer", default=30)},
        "schedule.list": empty,
        "schedule.cancel": {"job_id": _p("string"), "expected_version": _p("integer")},
        "schedule.run_due": {"worker_id": _p("string"), "limit": _p("integer", default=10)},
        "memory.remember": {"kind": _p("string"), "content": _p("string"), "source_checkpoints": _p("array"), "source_content_hash": _p("string"), "truth_status": _p("string"), "confidence": _p("number")},
        "memory.search": {"query": _p("string"), "include_history": _p("boolean", default=False), "limit": _p("integer", default=20)},
        "memory.list": {"include_history": _p("boolean", default=False), "limit": _p("integer", default=100)},
        "memory.update": {"memory_id": _p("string"), "content": _p("string"), "source_checkpoints": _p("array"), "source_content_hash": _p("string"), "truth_status": _p("string"), "confidence": _p("number")},
        "memory.forget": {"memory_id": _p("string")},
        "memory.clear": empty,
        "git.status": {"paths": _p("array", default=None)},
        "git.diff": {"paths": _p("array", default=None), "staged": _p("boolean", default=False), "revision": _p("string", default=None)},
        "git.log": {"max_count": _p("integer", default=20), "paths": _p("array", default=None)},
        "git.branch_info": empty,
        "git.create_checkpoint": {"message": _p("string"), "paths": _p("array")},
        "git.revert_commit": {"commit": _p("string"), "paths": _p("array")},
        "git.restore_paths": {"paths": _p("array"), "source": _p("string", default="HEAD")},
        "workflow.list": empty,
        "workflow.run": {"workflow_id": _p("string"), "workflow_version": _p("string"), "inputs": _p("object")},
        "capability.list": empty,
        "capability.describe": {"capability_id": _p("string")},
        "capability.acquire": {"gap_id": _p("string"), "proposal": _p("object")},
    }


_FIELDS = _contract_fields()
_INPUT_MODELS: dict[str, type[BaseModel]] = {
    name: create_model(
        "".join(part.title() for part in name.replace("_", ".").split(".")) + "Input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    for name, fields in _FIELDS.items()
}
_INPUT_MODELS.update(
    {
        "file.read": FileReadInput,
        "file.read_lines": FileReadLinesInput,
        "file.list": FileListInput,
        "file.search": FileSearchInput,
        "file.info": FileInfoInput,
        "terminal.run": TerminalRunInput,
        "terminal.status": ProcessInput,
        "terminal.cancel": ProcessInput,
        "browser.open": BrowserOpenInput,
        "browser.follow_link": BrowserFollowInput,
        "browser.extract": BrowserExtractInput,
        "browser.screenshot": BrowserScreenshotInput,
        "web.search": WebSearchInput,
        "web.fetch": WebFetchInput,
        "web.download": WebDownloadInput,
        "schedule.create": ScheduleCreateInput,
        "schedule.list": EmptyInput,
        "schedule.cancel": ScheduleCancelInput,
        "memory.remember": MemoryRememberInput,
        "memory.search": MemorySearchInput,
        "memory.list": MemoryListInput,
        "memory.update": MemoryUpdateInput,
        "memory.forget": MemoryIdInput,
        "memory.clear": EmptyInput,
        "git.status": GitPathsInput,
        "git.diff": GitDiffInput,
        "git.log": GitLogInput,
        "git.branch_info": EmptyInput,
        "git.create_checkpoint": GitCheckpointInput,
        "git.revert_commit": GitRevertInput,
        "git.restore_paths": GitRestoreInput,
        "capability.list": EmptyInput,
        "capability.describe": CapabilityDescribeInput,
    }
)


_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "file.read": ("path",), "file.read_lines": ("path",), "file.write": ("path",),
    "file.append": ("path",), "file.list": ("path",), "file.search": ("path",),
    "file.info": ("path",), "file.mkdir": ("path",), "file.copy": ("source", "destination"),
    "file.move": ("source", "destination"), "file.delete": ("path",),
    "terminal.run": ("cwd",), "browser.screenshot": ("target_path",),
    "web.download": ("target_path",),
}

_READ_ONLY = {
    "file.read", "file.read_lines", "file.list", "file.search", "file.info",
    "terminal.status", "browser.open", "browser.follow_link", "browser.extract",
    "web.search", "web.fetch", "schedule.list", "memory.search", "memory.list",
    "git.status", "git.diff", "git.log", "git.branch_info", "workflow.list",
    "capability.list", "capability.describe",
}
_CONFIRMED = {
    "file.write", "file.append", "file.copy", "file.move", "file.delete",
    "terminal.cancel", "web.download", "schedule.cancel", "memory.clear",
    "git.create_checkpoint", "git.revert_commit", "git.restore_paths", "capability.acquire",
}
_MEMORY = {name for name in CORE_TOOL_NAMES if name.startswith("memory.")}
_PROJECT = {
    name for name in CORE_TOOL_NAMES
    if name.startswith(("git.", "schedule.", "workflow.", "capability."))
}

_METHODS = {
    name: name.replace(".", "_")
    for name in BASIC_TOOL_NAMES
}

_SPACE_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "space.check_environment": SpaceEnvironmentInput,
    "space.convert_time": SpaceTimeInput,
    "space.transform_frame": SpaceFrameInput,
    "space.convert_orbit_representation": SpaceOrbitRepresentationInput,
    "space.propagate_orbit": SpacePropagationInput,
    "space.cross_validate_results": SpaceCrossValidationInput,
}


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string"},
            "result": {"type": "object"},
            "error": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "audit_id": {"type": "string"},
            "operation_id": {"type": "string"},
            "recovery_class": {"type": "string"},
        },
        "required": ["status", "result", "audit_id", "operation_id", "recovery_class"],
    }


class CoreToolCatalog:
    def __init__(self, entries: list[ToolCatalogEntry], bindings: Mapping[str, _Binding]):
        self._entries = tuple(entries)
        self._by_name = {entry.tool_name: entry for entry in entries}
        self._bindings = dict(bindings)
        if tuple(self._by_name) != CORE_TOOL_NAMES:
            raise ValueError("core tool inventory is not canonical")
        available = {entry.tool_name for entry in entries if entry.manifest.status == "available"}
        if available != set(self._bindings):
            raise ValueError("available manifests and executable bindings must have exact parity")
        for binding in self._bindings.values():
            if not binding.entrypoint.startswith(
                (
                    "aerospace_agent.mcp.tools.core_tool_adapters.",
                    "aerospace_agent.mcp.tools.space_tools.",
                )
            ):
                raise ValueError("tool binding is outside the current-repository adapter root")
            if Path(inspect.getsourcefile(binding.handler) or "").resolve() != binding.adapter_path:
                raise ValueError("tool binding source does not match its adapter path")

    def entries(self) -> list[ToolCatalogEntry]:
        return [ToolCatalogEntry.model_validate(item.model_dump(mode="python")) for item in self._entries]

    def get(self, tool_name: str) -> ToolCatalogEntry:
        return ToolCatalogEntry.model_validate(self._by_name[tool_name].model_dump(mode="python"))

    def executable_names(self) -> tuple[str, ...]:
        return tuple(name for name in CORE_TOOL_NAMES if name in self._bindings)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.tool_name,
                "description": f"Agent Core {item.tool_name} tool.",
                "inputSchema": item.input_schema,
                "outputSchema": item.output_schema,
                "status": item.manifest.status,
                "availabilityReason": item.availability_reason,
                "capabilityId": item.manifest.capability_id,
            }
            for item in self._entries
        ]

    def _validated_bindings_for_bootstrap(self) -> tuple[_Binding, ...]:
        """Internal bootstrap view; never pass this view to routing/planning."""
        return tuple(self._bindings[name] for name in self.executable_names())

    def register_into(self, registry: ExecutionRegistry) -> tuple[str, ...]:
        if not isinstance(registry, ExecutionRegistry):
            raise TypeError("registry must be ExecutionRegistry")
        registered: list[str] = []
        for binding in self._validated_bindings_for_bootstrap():
            entry = self._by_name[binding.tool_name]
            registry.register(
                kind="tool",
                manifest=entry.manifest,
                executor_name=binding.tool_name,
                handler=binding.handler,
                input_model=binding.input_model,
                entrypoint=binding.entrypoint,
                adapter_path=binding.adapter_path,
                recovery_class=binding.recovery_class,
                path_fields=binding.path_fields,
                requires_confirmation=binding.requires_confirmation,
            )
            registered.append(binding.tool_name)
        return tuple(registered)


def build_core_tool_catalog(
    workspace_root: str | Path,
    services: CoreToolServices,
    *,
    project_id: str | None = None,
    thread_id: str | None = None,
) -> CoreToolCatalog:
    root = Path(workspace_root).resolve()
    adapter = CoreToolAdapters(services, project_id=project_id, thread_id=thread_id)
    adapter_path = Path(inspect.getsourcefile(CoreToolAdapters) or "").resolve()
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    availability = _availability(services, project_id=project_id, workspace_root=root)
    entries: list[ToolCatalogEntry] = []
    bindings: dict[str, _Binding] = {}
    for name in BASIC_TOOL_NAMES:
        available, reason = availability[name]
        method_name = _METHODS[name]
        handler = getattr(adapter, method_name, None) if available else None
        if available and handler is None:
            raise ValueError(f"available tool has no adapter: {name}")
        risk = "read_only" if name in _READ_ONLY else "high_risk" if name in _CONFIRMED else "project_write"
        category = "memory" if name in _MEMORY else "project" if name in _PROJECT else "basic"
        recovery = (
            "read_only"
            if name in _READ_ONLY
            else "reversible"
            if name.startswith("file.")
            else "manual_recovery"
        )
        requires_confirmation = name in _CONFIRMED or recovery == "manual_recovery"
        source = "aerospace_agent.mcp.tools.core_tool_adapters"
        manifest = CapabilityManifest(
            capability_id="core." + name,
            version="1.0.0",
            category=category,
            status="available" if available else "unavailable",
            intents=[name.split(".", 1)[0]],
            tool_names=[name],
            risk_level=risk,
            source=source,
        )
        input_model = _INPUT_MODELS[name]
        entries.append(
            ToolCatalogEntry(
                tool_name=name,
                manifest=manifest,
                input_schema=input_model.model_json_schema(),
                output_schema=_output_schema(),
                availability_reason=reason,
                requires_confirmation=requires_confirmation,
                recovery_class=recovery,
                path_fields=_PATH_FIELDS.get(name, ()),
                adapter_sha256=adapter_hash if available else None,
            )
        )
        if handler is not None:
            target = inspect.unwrap(handler.__func__ if inspect.ismethod(handler) else handler)
            bindings[name] = _Binding(
                tool_name=name,
                handler=handler,
                input_model=input_model,
                entrypoint=f"{target.__module__}.{target.__qualname__}",
                adapter_path=adapter_path,
                recovery_class=recovery,
                path_fields=_PATH_FIELDS.get(name, ()),
                requires_confirmation=requires_confirmation,
            )
    for spec in SPACE_TOOL_SPECS:
        name = spec["name"]
        available = spec["status"] == "available"
        handler = SPACE_TOOL_CALLS.get(name)
        if available != (handler is not None):
            raise ValueError("SpaceBasic manifest and executor parity mismatch")
        input_model = _SPACE_INPUT_MODELS.get(name)
        if available and input_model is None:
            raise ValueError(f"available SpaceBasic tool lacks an inspectable input model: {name}")
        if input_model is None:
            input_model = create_model(
                "UnavailableSpaceValidateStateInput",
                __config__=ConfigDict(extra="forbid"),
                state=(dict[str, Any], ...),
            )
        manifest = CapabilityManifest(
            capability_id="space_basic." + name.removeprefix("space."),
            version="1.0.0",
            category="space_basic",
            status="available" if available else "unavailable",
            intents=["space_basic", name.removeprefix("space.")],
            tool_names=[name],
            risk_level="read_only",
            source="aerospace_agent.mcp.tools.space_tools",
        )
        entries.append(
            ToolCatalogEntry(
                tool_name=name,
                manifest=manifest,
                input_schema=spec["input_schema"],
                output_schema=spec["output_schema"],
                availability_reason=(
                    "adapter and declared contract tests are present"
                    if available
                    else "dedicated adapter and contract test are absent"
                ),
                recovery_class="read_only",
                adapter_sha256=spec["adapter_sha256"] if available else None,
            )
        )
        if handler is not None:
            target = inspect.unwrap(handler)
            space_adapter_path = Path(inspect.getsourcefile(target) or "").resolve()
            bindings[name] = _Binding(
                tool_name=name,
                handler=handler,
                input_model=input_model,
                entrypoint=f"{target.__module__}.{target.__qualname__}",
                adapter_path=space_adapter_path,
                recovery_class="read_only",
                path_fields=(),
                requires_confirmation=False,
            )
    return CoreToolCatalog(entries, bindings)


def _availability(
    services: CoreToolServices,
    *,
    project_id: str | None,
    workspace_root: Path,
) -> dict[str, tuple[bool, str]]:
    result = {name: (False, "service or verified adapter is unavailable") for name in BASIC_TOOL_NAMES}
    if services.files is not None:
        if Path(services.files.root).resolve() != workspace_root:
            reason = "FileService root does not equal the catalog/ExecutionRegistry workspace"
            for name in (
                "file.read", "file.read_lines", "file.write", "file.append", "file.list",
                "file.search", "file.info", "file.mkdir", "file.copy", "file.move", "file.delete",
            ):
                result[name] = (False, reason)
        else:
            for name in ("file.read", "file.read_lines", "file.list", "file.search", "file.info"):
                result[name] = (True, "validated workspace-bound FileService read adapter")
            journal = getattr(services.files, "journal", None)
            fixed = (
                getattr(journal, "database_path", None)
                == workspace_root / ".agent_core" / "operation_journal.sqlite3"
                and getattr(journal, "backup_dir", None)
                == workspace_root / ".agent_core" / "preimages"
            )
            for name in ("file.write", "file.append", "file.mkdir", "file.copy", "file.move", "file.delete"):
                result[name] = (
                    (True, "journal-bound reversible FileService adapter")
                    if fixed
                    else (False, "FileService does not use the fixed workspace journal/backup paths")
                )
    if services.terminal is not None:
        for name in ("terminal.run", "terminal.status", "terminal.cancel"):
            result[name] = (True, "validated TerminalService adapter")
    if services.browser is not None:
        for name in ("browser.open", "browser.follow_link", "browser.extract"):
            result[name] = (True, "validated read-only BrowserService adapter")
        if getattr(services.browser, "_screenshot_adapter", None) is not None:
            result["browser.screenshot"] = (True, "validated screenshot adapter")
    if services.web is not None:
        result["web.fetch"] = (True, "validated public-web fetch adapter")
        result["web.download"] = (True, "confirmation-gated public-web download adapter")
        if getattr(services.web, "search_endpoint", None) or getattr(
            services.web, "search_providers", ()
        ):
            result["web.search"] = (True, "configured public-web search provider")
    if services.scheduler is not None and project_id:
        for name in ("schedule.create", "schedule.list", "schedule.cancel"):
            result[name] = (True, "validated namespace-bound SchedulerService adapter")
        result["schedule.run_due"] = (False, "no end-to-end due-job workflow executor is bound")
    if services.memory is not None:
        for name in ("memory.remember", "memory.search", "memory.list", "memory.update", "memory.forget", "memory.clear"):
            result[name] = (True, "validated thread-bound SessionMemoryService adapter")
    if services.git is not None:
        probe = services.git.availability()
        if probe.status == "success":
            for name in ("git.status", "git.diff", "git.log", "git.branch_info", "git.create_checkpoint", "git.revert_commit", "git.restore_paths"):
                result[name] = (True, "Git executable and repository probe passed")
        else:
            reason = probe.error.message if probe.error else "Git repository is unavailable"
            for name in ("git.status", "git.diff", "git.log", "git.branch_info", "git.create_checkpoint", "git.revert_commit", "git.restore_paths"):
                result[name] = (False, reason)
    if services.capabilities is not None:
        result["capability.list"] = (True, "manifest-only CapabilityRegistry adapter")
        result["capability.describe"] = (True, "manifest-only CapabilityRegistry adapter")
    # workflow.run needs an immutable snapshot plus ExecutionService orchestration;
    # capability.acquire needs an approved acquisition service. A registry alone
    # is not an executor for either operation.
    return result


__all__ = [
    "CORE_TOOL_NAMES",
    "BASIC_TOOL_NAMES",
    "SPACE_BASIC_TOOL_NAMES",
    "CoreToolCatalog",
    "CoreToolServices",
    "ToolCatalogEntry",
    "build_core_tool_catalog",
]
