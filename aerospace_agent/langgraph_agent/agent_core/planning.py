"""Immutable TaskPlan construction and exact execution-step binding."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from .models import CapabilitySnapshot, TaskPlan, WorkflowSnapshot


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compute_task_plan_sha256(plan: TaskPlan | Mapping[str, Any]) -> str:
    data = (
        plan.model_dump(mode="json")
        if isinstance(plan, TaskPlan)
        else dict(plan)
    )
    data.pop("plan_sha256", None)
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def build_task_plan(payload: Mapping[str, Any]) -> TaskPlan:
    data = dict(payload)
    data["plan_sha256"] = "0" * 64
    provisional = TaskPlan.model_validate(data)
    data["plan_sha256"] = compute_task_plan_sha256(provisional)
    return TaskPlan.model_validate(data)


def _step_binding(step) -> tuple[str, str]:
    if step.executor_type in {"basic_tool", "space_basic_tool"}:
        return "tool", str(step.tool_name)
    if step.executor_type == "workflow":
        return "workflow", str(step.workflow_id)
    if step.executor_type == "domain_subgraph":
        return "domain", str(step.domain_subgraph)
    if step.executor_type == "capability_builder":
        return "capability_builder", str(step.capability_gap_id)
    return "human", step.capability


class PlanExecutionVerifier:
    """SQLite-backed verifier for exact plan namespace and step ownership."""

    _SCHEMA_VERSION = 1

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported plan schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE task_plans (
                        plan_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        plan_sha256 TEXT NOT NULL UNIQUE,
                        plan_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE plan_step_bindings (
                        plan_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        capability_id TEXT NOT NULL,
                        executor_name TEXT NOT NULL,
                        PRIMARY KEY(plan_id, step_id),
                        FOREIGN KEY(plan_id) REFERENCES task_plans(plan_id)
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def register_plan(self, plan: TaskPlan) -> None:
        checked = TaskPlan.model_validate(plan.model_dump(mode="python"))
        if compute_task_plan_sha256(checked) != checked.plan_sha256:
            raise ValueError("TaskPlan canonical hash mismatch")
        plan_json = _canonical_json(checked)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT plan_sha256 FROM task_plans WHERE plan_id = ?",
                (checked.plan_id,),
            ).fetchone()
            if existing is not None:
                if existing["plan_sha256"] != checked.plan_sha256:
                    connection.rollback()
                    raise ValueError("TaskPlan plan_id is immutable")
                connection.rollback()
                return
            connection.execute(
                """
                INSERT INTO task_plans(
                    plan_id, project_id, thread_id, root_run_id, plan_sha256, plan_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.plan_id,
                    checked.project_id,
                    checked.thread_id,
                    checked.root_run_id,
                    checked.plan_sha256,
                    plan_json,
                ),
            )
            for step in checked.steps:
                kind, executor = _step_binding(step)
                connection.execute(
                    """
                    INSERT INTO plan_step_bindings(
                        plan_id, step_id, kind, capability_id, executor_name
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (checked.plan_id, step.step_id, kind, step.capability, executor),
                )
            connection.commit()

    def has_plan_for_run(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        root_run_id: str,
    ) -> bool:
        """Return whether the persisted run is governed by an immutable plan."""

        if thread_id is None:
            return False
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM task_plans
                WHERE project_id = ? AND thread_id = ? AND root_run_id = ?
                LIMIT 1
                """,
                (project_id, thread_id, root_run_id),
            ).fetchone()
        return row is not None

    def verify(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        root_run_id: str,
        plan_id: str,
        plan_sha256: str,
        step_id: str,
        kind: str,
        capability_id: str,
        executor_name: str,
        arguments: Mapping[str, Any],
        capability_snapshot: CapabilitySnapshot,
        workflow_snapshot: WorkflowSnapshot | None,
        registry_snapshot_sha256: str,
        domain_state: Mapping[str, Any] | None = None,
    ) -> bool:
        if thread_id is None:
            return False
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT plan_json FROM task_plans
                WHERE plan_id = ? AND plan_sha256 = ?
                  AND project_id = ? AND thread_id = ? AND root_run_id = ?
                """,
                (
                    plan_id,
                    plan_sha256,
                    project_id,
                    thread_id,
                    root_run_id,
                ),
            ).fetchone()
        if row is None:
            return False
        try:
            plan = TaskPlan.model_validate(json.loads(row["plan_json"]))
        except (ValueError, TypeError, json.JSONDecodeError):
            return False
        snapshot = plan.execution_snapshot
        if registry_snapshot_sha256 != snapshot.registry_snapshot_sha256:
            return False
        if capability_snapshot not in snapshot.capability_snapshots:
            return False
        step = next((item for item in plan.steps if item.step_id == step_id), None)
        if step is None:
            handoff = next(
                (
                    item
                    for item in plan.handoffs
                    if item.conversion is not None
                    and step_id
                    == f"handoff-conversion:{item.source_step_id}:{item.target_step_id}"
                ),
                None,
            )
            if handoff is None or handoff.conversion is None:
                return False
            expected_argument_names = set(handoff.conversion.input_mapping.values())
            state = dict(domain_state or {})
            checkpoint_ref = state.get("checkpoint_ref")
            if (
                kind != "domain"
                or not executor_name
                or capability_id != handoff.conversion.converter_capability
                or capability_snapshot.capability_id != capability_id
                or set(arguments) != expected_argument_names
                or workflow_snapshot is not None
                or state.get("execution_role") != "handoff_conversion"
                or state.get("project_id") != project_id
                or state.get("thread_id") != thread_id
                or state.get("root_run_id") != root_run_id
                or state.get("plan_id") != plan_id
                or state.get("plan_sha256") != plan_sha256
                or state.get("step_id") != step_id
                or state.get("source_step_id") != handoff.source_step_id
                or state.get("target_step_id") != handoff.target_step_id
                or state.get("handoff") != handoff.model_dump(mode="json")
                or state.get("capability_snapshot")
                != capability_snapshot.model_dump(mode="json")
                or not isinstance(checkpoint_ref, dict)
                or checkpoint_ref.get("project_id") != project_id
                or checkpoint_ref.get("thread_id") != thread_id
                or not str(checkpoint_ref.get("checkpoint_id", "")).startswith("dag:")
            ):
                return False
            return True
        with closing(self._connect()) as connection:
            binding = connection.execute(
                """
                SELECT 1 FROM plan_step_bindings
                WHERE plan_id=? AND step_id=? AND kind=?
                  AND capability_id=? AND executor_name=?
                """,
                (plan_id, step_id, kind, capability_id, executor_name),
            ).fetchone()
        if binding is None or _canonical_json(step.inputs) != _canonical_json(dict(arguments)):
            return False
        if kind == "workflow":
            if workflow_snapshot is None or workflow_snapshot not in snapshot.workflow_snapshots:
                return False
        elif workflow_snapshot is not None:
            return False
        return True


__all__ = [
    "PlanExecutionVerifier",
    "build_task_plan",
    "compute_task_plan_sha256",
]
