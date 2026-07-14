from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

from aerospace_agent.langgraph_agent.agent_core.execution import (
    AuthorizedExecutor,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.execution_checkpoints import (
    ExecutionCheckpointStore,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityManifest,
    GoalBoundary,
    PlanExecutionSnapshot,
    PlanStep,
    VerificationCheck,
)
from aerospace_agent.langgraph_agent.agent_core.planning import (
    PlanExecutionVerifier,
    build_task_plan,
)


def _load_adapter(tmp_path: Path):
    adapter = tmp_path / "aerospace_agent" / "domains" / "test_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        """from pydantic import BaseModel, ConfigDict

class EmptyInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

def returns_bare_dict():
    return {'value': 42}

def returns_empty_artifacts():
    return {'artifacts': [], 'observation': 'nothing produced'}

class ArtifactInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    artifact: dict

def returns_artifact(artifact, state):
    concrete = dict(artifact)
    concrete['source_capability'] = state['capability_snapshot']
    concrete['source_checkpoints'] = [state['checkpoint_ref']]
    return {'artifacts': [concrete], 'observation': 'validated domain output'}
""",
        encoding="utf-8",
    )
    name = "aerospace_agent.domains.test_adapter"
    spec = importlib.util.spec_from_file_location(name, adapter)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, adapter


def _execute_domain(
    tmp_path: Path,
    handler_name: str,
    *,
    arguments: dict | None = None,
    domain_state: dict | None = None,
):
    module, adapter = _load_adapter(tmp_path)
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = ExecutionRegistry(
        tmp_path,
        audit_database_path=tmp_path / "audit.sqlite",
        plan_execution_verifier=verifier,
    )
    manifest = CapabilityManifest(
        capability_id="test-domain",
        version="1.0.0",
        category="domain",
        status="available",
        intents=["test-domain"],
        tool_names=[],
        risk_level="read_only",
        source="aerospace_agent.domains.test_adapter",
    )
    handler = getattr(module, handler_name)
    registry.register(
        kind="domain",
        manifest=manifest,
        executor_name=handler_name,
        handler=handler,
        input_model=module.ArtifactInput if handler_name == "returns_artifact" else module.EmptyInput,
        entrypoint=f"aerospace_agent.domains.test_adapter.{handler_name}",
        adapter_path=adapter,
        recovery_class="read_only",
    )
    arguments = dict(arguments or {})
    plan = build_task_plan(
        {
            "plan_id": "plan-1",
            "project_id": "project-1",
            "thread_id": "thread-1",
            "root_run_id": "run-1",
            "goal": GoalBoundary(
                objective="exercise domain boundary",
                in_scope=["test-domain"],
                success_criteria=["structured output"],
            ),
            "steps": [
                PlanStep(
                    step_id="step-1",
                    title="domain",
                    description="domain boundary",
                    executor_type="domain_subgraph",
                    capability="test-domain",
                    domain_subgraph=handler_name,
                    inputs=arguments,
                    expected_outputs=["artifact"],
                    verification=[
                        VerificationCheck(
                            check_id="schema",
                            description="domain output",
                            method="schema",
                            acceptance_rule="valid DomainExecutionOutput",
                        )
                    ],
                )
            ],
            "execution_snapshot": PlanExecutionSnapshot(
                capability_snapshots=[registry.snapshot("test-domain")],
                registry_snapshot_sha256="c" * 64,
                captured_at="2026-07-13T00:00:00+00:00",
            ),
            "created_at": "2026-07-13T00:00:00+00:00",
        }
    )
    verifier.register_plan(plan)
    request_data = dict(
        kind="domain",
        capability_id="test-domain",
        executor_name=handler_name,
        operation_id=f"op-{handler_name}",
        arguments=arguments,
        origin="planned",
        step_id="step-1",
    )
    if domain_state is not None:
        request_data["domain_state"] = {
            **domain_state,
            "capability_snapshot": registry.snapshot("test-domain").model_dump(mode="json"),
        }
    request = ExecutionRequest(**request_data)
    context = ExecutionContext(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        workspace_root=str(tmp_path),
        capability_snapshot=registry.snapshot("test-domain"),
        plan_id=plan.plan_id,
        plan_sha256=plan.plan_sha256,
        registry_snapshot_sha256=plan.execution_snapshot.registry_snapshot_sha256,
    )
    authorized = registry.resolve(request, context)
    assert isinstance(authorized, AuthorizedExecutor), authorized
    return ExecutionService(
        registry,
        checkpoint_store=ExecutionCheckpointStore(tmp_path / "checkpoints.sqlite"),
    ).execute(authorized)


def test_domain_executor_rejects_bare_dict_instead_of_promoting_it_to_tool_success(
    tmp_path: Path,
) -> None:
    result = _execute_domain(tmp_path, "returns_bare_dict")

    assert result.status == "failed"
    assert result.error is not None
    assert "DomainExecutionOutput" in result.error.message


def test_domain_executor_rejects_valid_shape_without_concrete_artifacts(
    tmp_path: Path,
) -> None:
    result = _execute_domain(tmp_path, "returns_empty_artifacts")

    assert result.status == "failed"
    assert result.error is not None
    assert "artifact" in result.error.message.lower()


def test_domain_executor_validates_and_preserves_domain_output_with_bound_state(
    tmp_path: Path,
) -> None:
    content = json.dumps(
        {"position": [1.0, 2.0, 3.0], "epoch": "2026-07-13T00:00:00Z"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    payload = tmp_path / "data" / "langgraph" / "artifacts" / "sha256" / digest
    payload.parent.mkdir(parents=True)
    payload.write_bytes(content)
    artifact = {
        "artifact_id": "artifact-1",
        "payload_ref": {
            "uri": f"artifact://sha256/{digest}",
            "sha256": digest,
            "byte_length": len(content),
            "media_type": "application/json",
        },
        "schema_id": "orbit-state",
        "schema_version": "1.0",
        "metadata": {
            "quantity_units": {"position": "m"},
            "frame_id": "ICRF",
            "time_system": "UTC",
            "epoch_field": "epoch",
        },
    }
    result = _execute_domain(
        tmp_path,
        "returns_artifact",
        arguments={"artifact": artifact},
        domain_state={
            "checkpoint_ref": {
                "project_id": "project-1",
                "thread_id": "thread-1",
                "checkpoint_id": "cp-before",
            }
        },
    )

    assert result.status == "success"
    output = result.result["domain_output"]
    assert output["artifacts"][0]["artifact_id"] == "artifact-1"
    assert output["artifacts"][0]["source_checkpoints"][0]["checkpoint_id"] == "cp-before"
