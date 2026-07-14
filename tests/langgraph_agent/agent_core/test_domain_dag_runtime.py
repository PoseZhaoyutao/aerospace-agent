from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent_core.artifacts import ArtifactStore
from aerospace_agent.langgraph_agent.agent_core.dag import (
    CanonicalMetadataVocabulary,
    CheckpointedDAGExecutor,
)
from aerospace_agent.langgraph_agent.agent_core.execution import (
    ExecutionRegistry,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.execution_checkpoints import (
    ExecutionCheckpointStore,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    ArtifactRef,
    ArtifactSchemaManifest,
    CapabilityManifest,
    CheckResult,
    DomainDataMetadata,
    GoalBoundary,
    PlanExecutionSnapshot,
    PlanStep,
    VerificationCheck,
)
from aerospace_agent.langgraph_agent.agent_core.planning import (
    PlanExecutionVerifier,
    build_task_plan,
)
from aerospace_agent.langgraph_agent.agent_core.review import ReviewAssessment, ReviewService


def _load_adapter(tmp_path: Path):
    adapter = tmp_path / "aerospace_agent" / "domains" / "dag_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        """from pydantic import BaseModel, ConfigDict

SOURCE_CALLS = 0
TARGET_CALLS = 0
CONVERTER_CALLS = 0
LAST_TARGET_POSITION = None
CONVERTED_ARTIFACT = None

class DomainInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    artifact: dict
    position: list[float] | None = None

class ConversionInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    position: list[float]

def _output(artifact, state):
    concrete = dict(artifact)
    concrete['source_capability'] = state['capability_snapshot']
    concrete['source_checkpoints'] = [state['checkpoint_ref']]
    return {'artifacts': [concrete], 'observation': 'domain step complete'}

def execute_source(artifact, position=None, state=None):
    global SOURCE_CALLS
    SOURCE_CALLS += 1
    return _output(artifact, state)

def execute_target(artifact, position=None, state=None):
    global TARGET_CALLS, LAST_TARGET_POSITION
    TARGET_CALLS += 1
    LAST_TARGET_POSITION = position
    if position is None:
        raise ValueError('persisted handoff input was not supplied')
    return _output(artifact, state)

def execute_converter(position, state=None):
    global CONVERTER_CALLS
    CONVERTER_CALLS += 1
    if CONVERTED_ARTIFACT is None:
        raise ValueError('converted artifact fixture is not configured')
    return _output(CONVERTED_ARTIFACT, state)
""",
        encoding="utf-8",
    )
    name = "aerospace_agent.domains.dag_adapter"
    spec = importlib.util.spec_from_file_location(name, adapter)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, adapter


def _payload_ref(root: Path, position: list[float]) -> ArtifactRef:
    content = json.dumps(
        {"position": position, "epoch": "2026-07-13T00:00:00Z"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path = root / "data" / "langgraph" / "artifacts" / "sha256" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef(
        uri=f"artifact://sha256/{digest}",
        sha256=digest,
        byte_length=len(content),
        media_type="application/json",
    )


def _artifact_seed(
    root: Path,
    artifact_id: str,
    position: list[float],
    *,
    metadata: DomainDataMetadata | None = None,
) -> dict:
    return {
        "artifact_id": artifact_id,
        "payload_ref": _payload_ref(root, position).model_dump(mode="json"),
        "schema_id": "orbit-state",
        "schema_version": "1.0",
        "metadata": (
            metadata
            or DomainDataMetadata(
                quantity_units={"position": "m"},
                frame_id="ICRF",
                time_system="UTC",
                epoch_field="epoch",
            )
        ).model_dump(mode="json"),
    }


def _runtime(
    tmp_path: Path,
    *,
    target_position: list[float] | None = None,
    conversion: bool = False,
    conversion_tamper: str | None = None,
):
    module, adapter = _load_adapter(tmp_path)
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = ExecutionRegistry(
        tmp_path,
        audit_database_path=tmp_path / "audit.sqlite",
        plan_execution_verifier=verifier,
    )
    for capability_id, executor_name in (
        ("simulation.impl", "execute_source"),
        ("orbit.impl", "execute_target"),
        ("frames.impl", "execute_converter"),
    ):
        registry.register(
            kind="domain",
            manifest=CapabilityManifest(
                capability_id=capability_id,
                version="1.0.0",
                category="domain",
                status="available",
                intents=[capability_id],
                tool_names=[],
                risk_level="read_only",
                source="aerospace_agent.domains.dag_adapter",
            ),
            executor_name=executor_name,
            handler=getattr(module, executor_name),
            input_model=(
                module.ConversionInput
                if executor_name == "execute_converter"
                else module.DomainInput
            ),
            entrypoint=f"aerospace_agent.domains.dag_adapter.{executor_name}",
            adapter_path=adapter,
            recovery_class="read_only",
        )
    source_position = [1.0, 2.0, 3.0]
    converted_position = [0.001, 0.002, 0.003]
    target_position = target_position or (
        converted_position if conversion else source_position
    )
    source_metadata = DomainDataMetadata(
        quantity_units={"position": "m"},
        frame_id="ICRF",
        time_system="UTC",
        epoch_field="epoch",
    )
    target_metadata = (
        DomainDataMetadata(
            quantity_units={"position": "km"},
            frame_id="ITRF",
            time_system="TAI",
            epoch_field="epoch",
        )
        if conversion
        else source_metadata
    )
    if conversion:
        converted_artifact = _artifact_seed(
            tmp_path,
            "converted-artifact",
            converted_position,
            metadata=target_metadata,
        )
        if conversion_tamper == "metadata":
            converted_artifact["metadata"] = source_metadata.model_dump(mode="json")
        elif conversion_tamper == "hash":
            converted_artifact["payload_ref"]["sha256"] = "f" * 64
        module.CONVERTED_ARTIFACT = converted_artifact
    source = PlanStep(
        step_id="source-step",
        title="source",
        description="produce orbit artifact",
        executor_type="domain_subgraph",
        capability="simulation.impl",
        domain_subgraph="execute_source",
        inputs={
            "artifact": _artifact_seed(tmp_path, "source-artifact", source_position),
            "position": None,
        },
        expected_outputs=["position"],
        verification=[
            VerificationCheck(
                check_id="source-schema",
                description="source schema",
                method="schema",
                acceptance_rule="valid artifact",
            )
        ],
    )
    target = PlanStep(
        step_id="target-step",
        title="target",
        description="consume orbit artifact",
        dependencies=["source-step"],
        executor_type="domain_subgraph",
        capability="orbit.impl",
        domain_subgraph="execute_target",
        inputs={
            "artifact": _artifact_seed(
                tmp_path,
                "target-artifact",
                target_position,
                metadata=target_metadata,
            ),
            "position": target_position,
        },
        expected_outputs=["position"],
        verification=[
            VerificationCheck(
                check_id="handoff-check",
                description="cross-domain mapping",
                method="cross_validation",
                acceptance_rule="mapped values are exact",
            )
        ],
    )
    plan = build_task_plan(
        {
            "plan_id": "domain-plan",
            "project_id": "project-1",
            "thread_id": "thread-1",
            "root_run_id": "run-1",
            "goal": GoalBoundary(
                objective="cross-domain handoff",
                in_scope=["simulation", "orbit"],
                success_criteria=["validated target execution"],
            ),
            "steps": [source, target],
            "handoffs": [
                {
                    "source_step_id": "source-step",
                    "target_step_id": "target-step",
                    "source_domain": "execute_source",
                    "target_domain": "execute_target",
                    "reason": "state transfer",
                    "required_inputs": ["position"],
                    "expected_outputs": ["position"],
                    "source_output_mapping": {"position": "state.position"},
                    "target_input_mapping": {"state.position": "position"},
                    "source_metadata": source_metadata,
                    "target_metadata": target_metadata,
                    **(
                        {
                            "conversion": {
                                "converter_capability": "frames.impl",
                                "input_mapping": {"position": "position"},
                                "output_mapping": {"position": "position"},
                                "validation_check_id": "handoff-check",
                            }
                        }
                        if conversion
                        else {}
                    ),
                }
            ],
            "execution_snapshot": PlanExecutionSnapshot(
                capability_snapshots=[
                    registry.snapshot("simulation.impl"),
                    registry.snapshot("orbit.impl"),
                    *(
                        [registry.snapshot("frames.impl")]
                        if conversion
                        else []
                    ),
                ],
                registry_snapshot_sha256="c" * 64,
                captured_at="2026-07-13T00:00:00+00:00",
            ),
            "created_at": "2026-07-13T00:00:00+00:00",
        }
    )
    schema = ArtifactSchemaManifest(
        schema_id="orbit-state",
        schema_version="1.0",
        payload_json_schema={
            "type": "object",
            "required": ["position", "epoch"],
            "properties": {
                "position": {"type": "array", "minItems": 3, "maxItems": 3},
                "epoch": {"type": "string"},
            },
        },
        required_quantity_fields=["position"],
        requires_frame=True,
        requires_time_system=True,
        requires_epoch_field=True,
    )
    store = ArtifactStore(
        tmp_path,
        [schema],
        schema_approval_verifier=lambda _manifest: True,
        checkpoint_verifier=lambda _checkpoint: True,
        capability_snapshot_verifier=lambda snapshot: snapshot
        in plan.execution_snapshot.capability_snapshots,
    )
    dag = CheckpointedDAGExecutor(
        database_path=tmp_path / "dag.sqlite",
        workspace_root=tmp_path,
        registry=registry,
        execution_service=ExecutionService(
            registry,
            checkpoint_store=ExecutionCheckpointStore(tmp_path / "execution-checkpoints.sqlite"),
        ),
        plan_verifier=verifier,
        metadata_vocabulary=CanonicalMetadataVocabulary(
            quantity_units={"m", "km"},
            frame_ids={"ICRF", "ITRF"},
            time_systems={"UTC", "TAI"},
        ),
        artifact_store=store,
        handoff_validation_checks={"handoff-check": lambda _context: True},
    )
    return module, registry, dag, plan, store


def test_raw_converter_callback_injection_surface_is_absent() -> None:
    assert "handoff_converters" not in inspect.signature(
        CheckpointedDAGExecutor
    ).parameters


def test_schema_and_metadata_conversion_runs_through_authorized_execution(
    tmp_path: Path,
) -> None:
    module, registry, dag, plan, _store = _runtime(tmp_path, conversion=True)

    outcome = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert outcome.status == "completed", {
        key: (value.status, value.error.message if value.error else None)
        for key, value in outcome.step_results.items()
    }
    assert module.SOURCE_CALLS == module.CONVERTER_CALLS == module.TARGET_CALLS == 1
    assert module.LAST_TARGET_POSITION == [0.001, 0.002, 0.003]
    converter_audits = [
        row
        for row in registry.audit_records()
        if row["capability_id"] == "frames.impl"
    ]
    assert len(converter_audits) == 1
    assert converter_audits[0]["kind"] == "domain"
    assert converter_audits[0]["status"] == "success"


def test_unregistered_conversion_capability_blocks_before_any_handler(
    tmp_path: Path,
) -> None:
    module, registry, dag, plan, _store = _runtime(tmp_path, conversion=True)
    registry._registrations = {
        key: value
        for key, value in registry._registrations.items()
        if key[1] != "frames.impl"
    }

    outcome = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert outcome.status == "invalid_plan"
    assert module.SOURCE_CALLS == module.CONVERTER_CALLS == module.TARGET_CALLS == 0


@pytest.mark.parametrize("tamper", ["metadata", "hash"])
def test_invalid_conversion_artifact_blocks_target_without_calling_it(
    tmp_path: Path,
    tamper: str,
) -> None:
    module, _registry, dag, plan, _store = _runtime(
        tmp_path,
        conversion=True,
        conversion_tamper=tamper,
    )

    outcome = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert outcome.status == "failed"
    assert module.SOURCE_CALLS == module.CONVERTER_CALLS == 1
    assert module.TARGET_CALLS == 0


def test_real_domain_handoff_persists_refs_revalidates_target_and_resumes(
    tmp_path: Path,
) -> None:
    module, registry, dag, plan, store = _runtime(tmp_path)

    first = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )
    second = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert first.status == second.status == "completed", {
        "first": {
            key: (value.status, value.error.message if value.error else None)
            for key, value in first.step_results.items()
        },
        "second": {
            key: (value.status, value.error.message if value.error else None)
            for key, value in second.step_results.items()
        },
    }
    assert module.SOURCE_CALLS == module.TARGET_CALLS == 1
    assert second.reused_step_ids == ["source-step", "target-step"]
    source_refs = first.state.step_states[0].last_output_refs
    assert "artifact:source-artifact" in source_refs
    assert any(item.startswith("handoff:") for item in source_refs)
    assert first.step_results["source-step"].audit_id not in source_refs
    handoff_ids = store.handoff_ids_for_target(
        project_id=plan.project_id,
        plan_id=plan.plan_id,
        target_step_id="target-step",
    )
    assert len(handoff_ids) == 1
    assert dag.verify_state(plan, first.state, first.step_results)


def test_orphaned_exchange_is_reused_after_checkpoint_crash_without_duplicate_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _registry, dag, plan, store = _runtime(tmp_path)
    original = dag._record_checkpoint
    crashed = False

    def crash_once(*args, **kwargs):
        nonlocal crashed
        step = args[1]
        if not crashed and step.step_id == "source-step" and kwargs.get("phase") == "after":
            crashed = True
            assert store.handoff_ids_for_target(
                project_id=plan.project_id,
                plan_id=plan.plan_id,
                target_step_id="target-step",
            )
            raise RuntimeError("simulated crash after artifact transaction")
        return original(*args, **kwargs)

    monkeypatch.setattr(dag, "_record_checkpoint", crash_once)
    with pytest.raises(RuntimeError, match="simulated crash"):
        dag.execute(
            plan,
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
        )

    resumed = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )
    repeated = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert resumed.status == repeated.status == "completed"
    assert module.SOURCE_CALLS == 1
    assert module.TARGET_CALLS == 1
    assert repeated.reused_step_ids == ["source-step", "target-step"]


def test_persisted_hash_or_mapping_mismatch_blocks_target_without_calling_handler(
    tmp_path: Path,
) -> None:
    module, _registry, dag, plan, _store = _runtime(
        tmp_path, target_position=[9.0, 9.0, 9.0]
    )

    outcome = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert outcome.status == "failed", {
        key: (value.status, value.error.message if value.error else None)
        for key, value in outcome.step_results.items()
    }
    assert module.SOURCE_CALLS == 1
    assert module.TARGET_CALLS == 0
    assert "mapping" in (outcome.step_results["target-step"].error.message.lower())


def test_review_derives_domain_pass_from_persisted_artifacts_not_caller_claims(
    tmp_path: Path,
) -> None:
    _module, _registry, dag, plan, _store = _runtime(tmp_path)
    outcome = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )
    assert outcome.status == "completed" and outcome.state is not None
    evidence = [
        *[result.audit_id for result in outcome.step_results.values()],
        *[
            state.last_checkpoint_id
            for state in outcome.state.step_states
            if state.last_checkpoint_id is not None
        ],
    ]
    assessment = ReviewAssessment(
        goal_satisfied=True,
        boundary_compliant=True,
        constraints_satisfied=True,
        evidence_sufficient=True,
        tool_execution_safe=True,
        checks=[
            CheckResult(
                check_id=check_id,
                passed=True,
                severity="error",
                message="validated by persisted evidence",
                evidence_refs=evidence,
            )
            for check_id in ("source-schema", "handoff-check")
        ],
        domain_reviews=[],
        confidence=0.95,
    )

    review = ReviewService(dag).review(
        plan=plan,
        state=outcome.state,
        step_results=outcome.step_results,
        assessment=assessment,
    )

    assert review.status == "passed"
    assert {item.domain for item in review.domain_reviews} == {
        "execute_source",
        "execute_target",
    }
    assert all(item.status == "passed" for item in review.domain_reviews)
