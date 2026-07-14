"""Current-repository adapters for Agent Core services.

The adapter object is bootstrap-only: planners receive manifests, never these
bound methods.  ExecutionRegistry verifies the concrete method identity and is
the only consumer allowed to register a binding.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from aerospace_agent.langgraph_agent.agent_core.models import CheckpointRef


def _json(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return {"items": [_json_item(item) for item in value]}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _json_item(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


class CoreToolAdapters:
    """Thin namespace-bound adapters used by the execution bootstrap."""

    def __init__(
        self,
        services: Any,
        *,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.services = services
        self.file_service = getattr(services, "files", None)
        self.project_id = project_id
        self.thread_id = thread_id

    def file_read(self, **arguments: Any):
        return self.services.files.read(**arguments)

    def file_read_lines(self, **arguments: Any):
        return self.services.files.read_lines(**arguments)

    def file_list(self, **arguments: Any):
        return self.services.files.list(**arguments)

    def file_search(self, **arguments: Any):
        return self.services.files.search(**arguments)

    def file_info(self, **arguments: Any):
        return self.services.files.info(**arguments)

    def file_write(
        self, *, operation_id: str, confirmed: bool = False, **arguments: Any
    ):
        return self.services.files.write(
            operation_id=operation_id, confirmed=confirmed, **arguments
        )

    def file_append(
        self, *, operation_id: str, confirmed: bool = False, **arguments: Any
    ):
        return self.services.files.append(
            operation_id=operation_id, confirmed=confirmed, **arguments
        )

    def file_mkdir(self, *, operation_id: str, **arguments: Any):
        return self.services.files.mkdir(operation_id=operation_id, **arguments)

    def file_copy(
        self, *, operation_id: str, confirmed: bool = False, **arguments: Any
    ):
        return self.services.files.copy(
            operation_id=operation_id, confirmed=confirmed, **arguments
        )

    def file_move(
        self, *, operation_id: str, confirmed: bool = False, **arguments: Any
    ):
        return self.services.files.move(
            operation_id=operation_id, confirmed=confirmed, **arguments
        )

    def file_delete(
        self, *, operation_id: str, confirmed: bool = False, **arguments: Any
    ):
        return self.services.files.delete(
            operation_id=operation_id, confirmed=confirmed, **arguments
        )

    def terminal_run(self, *, confirmed: bool = False, **arguments: Any):
        # TerminalService independently rejects writing/unknown/background
        # commands when no argument-bound confirmation proof is available.
        return self.services.terminal.run(confirmed=confirmed, **arguments)

    def terminal_status(self, **arguments: Any):
        return self.services.terminal.status(**arguments)

    def terminal_cancel(self, *, confirmed: bool = False, **arguments: Any):
        return self.services.terminal.cancel(confirmed=confirmed, **arguments)

    def browser_open(self, **arguments: Any):
        return self.services.browser.open(**arguments)

    def browser_follow_link(self, **arguments: Any):
        return self.services.browser.follow_link(**arguments)

    def browser_extract(self, **arguments: Any):
        return self.services.browser.extract(**arguments)

    def browser_screenshot(self, **arguments: Any):
        return self.services.browser.screenshot(**arguments)

    def web_search(self, **arguments: Any):
        return self.services.web.search(**arguments)

    def web_fetch(self, **arguments: Any):
        return self.services.web.fetch(**arguments)

    def web_download(self, *, confirmed: bool = False, **arguments: Any):
        return self.services.web.download(confirmed=confirmed, **arguments)

    def schedule_create(self, **arguments: Any) -> dict[str, Any]:
        kind = arguments.pop("kind")
        if kind == "reminder":
            job = self.services.scheduler.create_reminder(
                project_id=self._project(),
                thread_id=self.thread_id,
                due_at=arguments["due_at"],
                message=arguments.get("message"),
            )
        elif kind == "workflow":
            job = self.services.scheduler.create_workflow(
                project_id=self._project(),
                thread_id=self.thread_id,
                due_at=arguments["due_at"],
                workflow_id=arguments.get("workflow_id"),
                workflow_version=arguments.get("workflow_version"),
                inputs=arguments.get("inputs"),
                max_retries=arguments.get("max_retries", 0),
                retry_delay_seconds=arguments.get("retry_delay_seconds", 30),
            )
        else:
            raise ValueError("schedule.create kind must be reminder or workflow")
        return _json(job)

    def schedule_list(self, **arguments: Any) -> dict[str, Any]:
        jobs = self.services.scheduler.list_jobs(
            project_id=self._project(), thread_id=self.thread_id, **arguments
        )
        return {"jobs": [_json_item(item) for item in jobs]}

    def schedule_cancel(self, **arguments: Any) -> dict[str, Any]:
        job = self.services.scheduler.cancel(**arguments)
        return {"job": _json_item(job) if job is not None else None}

    def memory_remember(self, **arguments: Any) -> dict[str, Any]:
        arguments["source_checkpoints"] = self._checkpoints(arguments["source_checkpoints"])
        return _json(self.services.memory.remember(**arguments))

    def memory_search(self, **arguments: Any) -> dict[str, Any]:
        return {"memories": [_json_item(item) for item in self.services.memory.search(**arguments)]}

    def memory_list(self, **arguments: Any) -> dict[str, Any]:
        return {"memories": [_json_item(item) for item in self.services.memory.list(**arguments)]}

    def memory_update(self, **arguments: Any) -> dict[str, Any]:
        arguments["source_checkpoints"] = self._checkpoints(arguments["source_checkpoints"])
        return _json(self.services.memory.update(**arguments))

    def memory_forget(self, **arguments: Any) -> dict[str, Any]:
        return _json(self.services.memory.forget(**arguments))

    def memory_clear(
        self, *, confirmation_consumed: bool = False, **arguments: Any
    ) -> dict[str, Any]:
        if arguments:
            raise ValueError("memory.clear does not accept arguments")
        return {
            "cleared": self.services.memory.clear(
                confirmation_consumed=confirmation_consumed
            )
        }

    def git_status(self, **arguments: Any):
        return self.services.git.status(**arguments)

    def git_diff(self, **arguments: Any):
        return self.services.git.diff(**arguments)

    def git_log(self, **arguments: Any):
        return self.services.git.log(**arguments)

    def git_branch_info(self, **arguments: Any):
        return self.services.git.branch_info(**arguments)

    def git_create_checkpoint(
        self, *, confirmation_consumed: bool = False, **arguments: Any
    ):
        return self.services.git.create_checkpoint(
            confirmation_consumed=confirmation_consumed, **arguments
        )

    def git_revert_commit(
        self, *, confirmation_consumed: bool = False, **arguments: Any
    ):
        return self.services.git.revert_commit(
            confirmation_consumed=confirmation_consumed, **arguments
        )

    def git_restore_paths(
        self, *, confirmation_consumed: bool = False, **arguments: Any
    ):
        return self.services.git.restore_paths(
            confirmation_consumed=confirmation_consumed, **arguments
        )

    def capability_list(self, **arguments: Any) -> dict[str, Any]:
        if arguments:
            raise ValueError("capability.list does not accept arguments")
        return {
            "capabilities": [
                item.model_dump(mode="json")
                for item in self.services.capabilities.list_manifests()
            ]
        }

    def capability_describe(self, **arguments: Any) -> dict[str, Any]:
        return _json(self.services.capabilities.get(arguments["capability_id"]))

    def _project(self) -> str:
        if not self.project_id:
            raise RuntimeError("project_id is required for this tool")
        return self.project_id

    @staticmethod
    def _checkpoints(values: Sequence[Mapping[str, Any] | CheckpointRef]) -> list[CheckpointRef]:
        return [
            item if isinstance(item, CheckpointRef) else CheckpointRef.model_validate(item)
            for item in values
        ]


__all__ = ["CoreToolAdapters"]
