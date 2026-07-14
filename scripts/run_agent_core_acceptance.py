#!/usr/bin/env python3
"""Run the Agent Core acceptance suite in one isolated artifact directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


ACCEPTANCE_ITEMS = (
    ("AC-01", "30+ self-contained tools and sole authorized execution boundary"),
    ("AC-02", "bounded relevant routing; conversation invokes no tools or RAG"),
    ("AC-03", "conditional one-budget RAG with negative request precedence"),
    ("AC-04", "direct simple execution and immutable complex TaskPlan snapshot"),
    ("AC-05", "invalid DAG/domain/metadata/handoff execution is rejected"),
    ("AC-06", "checkpoint resume, idempotency, confirmation and snapshot drift"),
    ("AC-07", "ReviewResult identity and non-success completion gates"),
    ("AC-08", "durable strict project/thread memory isolation"),
    ("AC-09", "idempotent explicit project initialization and legacy behavior"),
    ("AC-10", "internal scheduler lease/cancel/retry/approval invariants"),
    ("AC-11", "path, shell, download, Git and long-running confirmation safety"),
    ("AC-12", "truthful reversible, compensatable and manual recovery"),
    ("AC-13", "unapproved workflow/skill candidates cannot route or execute"),
    ("AC-14", "signed capability acquisition and hash-drift invalidation"),
    ("AC-15", "RAG crash recovery and generated-data hygiene"),
)

# Each design section 13 acceptance criterion is bound to tests that exercise
# that criterion.  Function selectors intentionally expand to every collected
# parameterized case; a selector that collects no case fails closed.
ACCEPTANCE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "AC-01": (
        "tests/langgraph_agent/agent_core/tools/test_tool_catalog.py::test_inventory_is_stable_self_contained_and_has_at_least_thirty_tools",
        "tests/langgraph_agent/agent_core/tools/test_tool_catalog.py::test_catalog_registers_only_available_handlers_in_execution_registry",
        "tests/langgraph_agent/agent_core/test_execution.py::test_resolve_returns_sealed_non_callable_and_only_service_invokes_handler",
    ),
    "AC-02": (
        "tests/langgraph_agent/agent_core/test_capabilities.py::test_candidates_are_relevant_available_and_hard_limited_to_twelve",
        "tests/langgraph_agent/agent_core/test_integration.py::test_default_core_conversation_uses_neither_tool_nor_rag_and_complex_is_honest_partial",
        "tests/langgraph_agent/agent_core/test_prompts.py::test_model_vendor_self_identification_is_replaced_by_aerospace_identity",
    ),
    "AC-03": (
        "tests/langgraph_agent/agent_core/test_rag_gate.py::test_private_rag_uses_only_three_positive_triggers_and_negative_override",
        "tests/langgraph_agent/agent_core/test_rag_gate.py::test_atomic_claim_allows_only_one_parallel_branch",
        "tests/langgraph_agent/agent_core/test_integration.py::test_core_rag_denial_wins_and_positive_request_consumes_one_root_budget",
    ),
    "AC-04": (
        "tests/langgraph_agent/agent_core/test_integration.py::test_default_core_file_read_crosses_authorized_executor_boundary",
        "tests/langgraph_agent/agent_core/test_planning.py::test_build_plan_computes_canonical_hash_and_verifier_persists_exact_binding",
        "tests/langgraph_agent/agent_core/test_models.py::test_task_plan_contains_no_mutable_step_status",
    ),
    "AC-05": (
        "tests/langgraph_agent/agent_core/test_dag.py::test_cycle_is_rejected_without_checkpoint_or_execution",
        "tests/langgraph_agent/agent_core/test_artifacts.py::test_store_fails_closed_on_uri_schema_metadata_snapshot_and_checkpoint",
        "tests/langgraph_agent/agent_core/test_artifacts.py::test_handoff_resolves_real_source_and_conversion_output_and_checks_metadata",
        "tests/langgraph_agent/test_domain_interfaces.py::test_domain_placeholders_return_only_capability_gaps_and_no_execution_objects",
        "tests/langgraph_agent/agent_core/test_domain_dag_runtime.py::test_raw_converter_callback_injection_surface_is_absent",
        "tests/langgraph_agent/agent_core/test_domain_dag_runtime.py::test_schema_and_metadata_conversion_runs_through_authorized_execution",
        "tests/langgraph_agent/agent_core/test_domain_dag_runtime.py::test_invalid_conversion_artifact_blocks_target_without_calling_it",
        "tests/langgraph_agent/agent_core/test_domain_dag_runtime.py::test_persisted_hash_or_mapping_mismatch_blocks_target_without_calling_handler",
    ),
    "AC-06": (
        "tests/langgraph_agent/agent_core/test_dag.py::test_dag_checkpoints_before_and_after_and_reuses_exact_idempotency_key",
        "tests/langgraph_agent/agent_core/test_dag.py::test_interrupted_write_is_not_retried_until_audited_state_inspection",
        "tests/langgraph_agent/agent_core/test_confirmation.py::test_confirmation_consumption_atomically_persists_continuation_checkpoint",
        "tests/langgraph_agent/agent_core/test_execution.py::test_snapshot_mismatch_blocks_before_invocation",
        "tests/langgraph_agent/agent_core/test_domain_dag_runtime.py::test_orphaned_exchange_is_reused_after_checkpoint_crash_without_duplicate_execution",
    ),
    "AC-07": (
        "tests/langgraph_agent/agent_core/test_review.py::test_review_passes_only_with_exact_identity_and_all_completion_gates",
        "tests/langgraph_agent/agent_core/test_review.py::test_partial_or_unsupported_work_cannot_be_declared_complete",
        "tests/langgraph_agent/agent_core/test_review.py::test_failed_domain_or_boundary_review_cannot_pass",
    ),
    "AC-08": (
        "tests/langgraph_agent/agent_core/test_session_memory.py::test_same_thread_memory_survives_restart_and_other_thread_cannot_read",
        "tests/langgraph_agent/agent_core/test_session_memory.py::test_checkpoint_namespace_and_validator_are_enforced",
        "tests/langgraph_agent/agent_core/test_context_assembler.py::test_context_assembler_never_reads_other_thread_or_untraceable_memory",
        "tests/langgraph_agent/agent_core/test_project_identity.py::test_project_memory_search_is_project_scoped_and_reads_only_indexed_sources",
    ),
    "AC-09": (
        "tests/langgraph_agent/agent_core/test_project_identity.py::test_initialize_is_idempotent_preserves_user_files_and_creates_required_layout",
        "tests/langgraph_agent/agent_core/test_integration.py::test_uninitialized_project_keeps_legacy_graph_behavior",
        "tests/langgraph_agent/agent_core/test_project_cli.py::test_legacy_repl_status_does_not_implicitly_initialize_project",
    ),
    "AC-10": (
        "tests/langgraph_agent/agent_core/test_scheduler.py::test_workflow_job_locks_snapshot_masks_inputs_and_creates_no_rag_run_on_claim",
        "tests/langgraph_agent/agent_core/test_scheduler.py::test_claim_revalidates_payload_workflow_approval_and_automation_policy",
        "tests/langgraph_agent/agent_core/test_scheduler.py::test_atomic_optimistic_cancel_and_safe_point_recovery_class",
        "tests/langgraph_agent/agent_core/test_scheduler.py::test_retry_is_only_for_locked_safe_workflows_and_attempt_ids_increment",
        "tests/langgraph_agent/agent_core/test_scheduler.py::test_expired_lease_rejects_mark_running",
        "tests/langgraph_agent/agent_core/test_scheduler.py::test_expired_running_lease_rejects_terminal_worker_transition",
    ),
    "AC-11": (
        "tests/langgraph_agent/agent_core/tools/test_files.py::test_symlink_escape_is_rejected_for_reads_and_writes",
        "tests/langgraph_agent/agent_core/tools/test_terminal.py::test_rejects_shell_strings_metacharacters_and_non_allowlisted_commands",
        "tests/langgraph_agent/agent_core/tools/test_web.py::test_download_requires_confirmation_and_expected_hash_before_atomic_replace",
        "tests/langgraph_agent/agent_core/test_git_service.py::test_service_never_offers_or_runs_banned_destructive_git_operations",
        "tests/langgraph_agent/agent_core/tools/test_terminal.py::test_git_diff_rejects_workspace_escape_in_every_path_argument_form",
    ),
    "AC-12": (
        "tests/langgraph_agent/agent_core/test_recovery.py::test_completed_file_operation_imports_full_evidence_and_rolls_back",
        "tests/langgraph_agent/agent_core/test_recovery.py::test_compensatable_operation_requires_external_approval_and_real_code_hashes",
        "tests/langgraph_agent/agent_core/test_recovery.py::test_manual_recovery_report_never_claims_restoration_or_offers_banned_git",
        "tests/langgraph_agent/agent_core/tools/test_files.py::test_multi_target_rollback_preflights_all_targets_before_restoring_any",
        "tests/langgraph_agent/agent_core/tools/test_files.py::test_multi_target_restore_failure_puts_every_postimage_back",
    ),
    "AC-13": (
        "tests/langgraph_agent/agent_core/test_evolution.py::test_workflow_candidate_is_not_an_active_workflow_before_approval",
        "tests/langgraph_agent/agent_core/test_capabilities.py::test_available_workflow_requires_external_approval_verifier",
        "tests/langgraph_agent/agent_core/test_execution_security.py::test_approval_is_rechecked_at_resolve_not_only_registration",
        "tests/langgraph_agent/agent_core/test_evolution.py::test_replay_failure_and_state_skips_are_rejected",
        "tests/langgraph_agent/agent_core/test_evolution.py::test_activation_recomputes_persistent_payload_and_does_not_consume_on_drift",
        "tests/langgraph_agent/test_evolution_service.py::test_agent_rejects_spoofed_boolean_evolution_approval",
    ),
    "AC-14": (
        "tests/langgraph_agent/agent_core/test_integrations.py::test_combined_digest_requires_external_ed25519_approval_for_all_import_roots",
        "tests/langgraph_agent/agent_core/test_integrations.py::test_lock_cache_local_code_and_evidence_boundaries_fail_closed",
        "tests/langgraph_agent/agent_core/test_integrations.py::test_manifest_or_evidence_drift_requires_a_new_approval",
        "tests/langgraph_agent/agent_core/test_integrations.py::test_one_build_per_capability_run_and_resume_exact_original_checkpoint",
        "tests/langgraph_agent/agent_core/test_execution_security.py::test_runtime_trust_baseline_is_rechecked_at_resolve_and_execute",
    ),
    "AC-15": (
        "tests/langgraph_agent/agent_core/test_rag_gate.py::test_crash_recovery_releases_unstarted_claim_but_never_retries_in_flight",
        "tests/langgraph_agent/agent_core/test_acceptance_runner.py::test_runner_uses_unique_artifact_root_writes_one_report_and_cleans_success",
        "tests/langgraph_agent/agent_core/test_data_hygiene.py::test_project_tree_has_no_stray_test_or_bytecode_artifacts",
    ),
}

OFFLINE_TARGETS = (
    "tests/langgraph_agent/agent_core",
    "tests/langgraph_agent/test_cli.py",
    "tests/langgraph_agent/test_context_cycle.py",
    "tests/langgraph_agent/test_graph_runtime.py",
    "tests/langgraph_agent/test_domain_interfaces.py",
    "tests/langgraph_agent/test_evolution_service.py",
    "tests/langgraph_agent/test_config.py",
    "tests/langgraph_agent/test_providers.py",
    "tests/langgraph_agent/test_runner.py",
    "tests/langgraph_agent/test_turns.py",
    "tests/test_space_tool_specs.py",
)

LIVE_QWEN_TARGET = "tests/langgraph_agent/test_qwen_acceptance.py"
LIVE_QWEN_EXPECTED_TESTS = 5


def _remove_runner_artifacts(path: Path) -> None:
    """Remove a runner-owned tree, including read-only integration fixtures."""

    def retry_readonly(function, target, error):
        if not isinstance(error, PermissionError):
            raise error
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(path, onexc=retry_readonly)


def _case_node_id(case: ET.Element) -> str:
    classname = case.attrib.get("classname", "").strip()
    name = case.attrib.get("name", "").strip()
    if not classname or not name:
        return ""
    return f"{classname.replace('.', '/')}.py::{name}"


def _case_status(case: ET.Element) -> str:
    if case.find("error") is not None:
        return "error"
    if case.find("failure") is not None:
        return "failed"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def _read_junit(path: Path) -> tuple[dict[str, str], str | None]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {}, f"junit_unavailable: {type(exc).__name__}: {exc}"
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        node_id = _case_node_id(case)
        if node_id:
            results[node_id] = _case_status(case)
    return results, None


def _summary(statuses: list[str]) -> dict[str, int]:
    return {
        name: statuses.count(name)
        for name in ("passed", "failed", "error", "skipped")
    }


def _matches_selector(node_id: str, selector: str) -> bool:
    return node_id == selector or node_id.startswith(selector + "[")


def _acceptance_results(test_results: dict[str, str]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for acceptance_id, description in ACCEPTANCE_ITEMS:
        selectors = ACCEPTANCE_EVIDENCE.get(acceptance_id, ())
        selected: dict[str, str] = {}
        missing: list[str] = [] if selectors else ["unmapped_acceptance"]
        for selector in selectors:
            matches = {
                node_id: outcome
                for node_id, outcome in test_results.items()
                if _matches_selector(node_id, selector)
            }
            if not matches:
                missing.append(selector)
            selected.update(matches)
        statuses = list(selected.values())
        passed = not missing and bool(statuses) and all(
            outcome == "passed" for outcome in statuses
        )
        if missing:
            item_return_code = 5
        elif passed:
            item_return_code = 0
        else:
            item_return_code = 1
        items.append(
            {
                "acceptance_id": acceptance_id,
                "description": description,
                "status": "passed" if passed else "failed",
                "evidence": {
                    "selectors": list(selectors),
                    "node_ids": sorted(selected),
                    "results": [
                        {"node_id": node_id, "status": selected[node_id]}
                        for node_id in sorted(selected)
                    ],
                    "return_code": item_return_code,
                    "summary": _summary(statuses),
                    "missing_selectors": missing,
                },
            }
        )
    return items


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]


def run_acceptance(
    workspace: str | Path,
    *,
    run_id: str | None = None,
    keep_artifacts: bool = False,
    live_qwen: bool = False,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    identifier = run_id or _run_id()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
        raise ValueError("run_id may contain only letters, digits, dot, underscore and hyphen")
    artifact_root = (root / ".test-artifacts" / "agent-core" / identifier).resolve()
    expected_parent = (root / ".test-artifacts" / "agent-core").resolve()
    if artifact_root.parent != expected_parent:
        raise ValueError("acceptance artifact root escaped its fixed workspace parent")
    artifact_root.mkdir(parents=True, exist_ok=False)

    junit_path = artifact_root / "offline-results.xml"
    pytest_root = artifact_root / "pytest"
    targets = list(OFFLINE_TARGETS)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(pytest_root),
        "--junitxml",
        str(junit_path),
    ]
    environment = dict(os.environ)
    environment["AEROSPACE_TEST_ARTIFACT_ROOT"] = str(artifact_root)
    environment["AGENT_CORE_LIVE_QWEN"] = "0"
    system_temp = artifact_root / "system-temp"
    system_temp.mkdir(parents=True, exist_ok=False)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TEMP"] = str(system_temp)
    environment["TMP"] = str(system_temp)
    environment["TMPDIR"] = str(system_temp)
    started_at = datetime.now(UTC)
    offline = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    test_results, junit_error = _read_junit(junit_path)
    acceptance_items = _acceptance_results(test_results)

    live_report: dict[str, object] = {
        "requested": live_qwen,
        "status": "not_run",
        "command": [],
        "return_code": None,
        "stdout": "",
        "stderr": "",
    }
    if live_qwen:
        live_junit = artifact_root / "live-qwen-results.xml"
        live_command = [
            sys.executable,
            "-m",
            "pytest",
            LIVE_QWEN_TARGET,
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(artifact_root / "live-qwen-pytest"),
            "--junitxml",
            str(live_junit),
        ]
        live_environment = dict(environment)
        live_environment["AGENT_CORE_LIVE_QWEN"] = "1"
        live = subprocess.run(
            live_command,
            cwd=root,
            env=live_environment,
            text=True,
            capture_output=True,
        )
        live_results, live_junit_error = _read_junit(live_junit)
        live_statuses = list(live_results.values())
        all_live_tests_passed = (
            len(live_statuses) == LIVE_QWEN_EXPECTED_TESTS
            and all(status == "passed" for status in live_statuses)
        )
        if live.returncode == 0 and all_live_tests_passed:
            live_status = "passed"
        elif live_statuses and all(status == "skipped" for status in live_statuses):
            live_status = "blocked"
        else:
            live_status = "failed"
        live_report = {
            "requested": True,
            "status": live_status,
            "command": live_command,
            "return_code": live.returncode,
            "stdout": live.stdout,
            "stderr": live.stderr,
            "summary": _summary(live_statuses),
            "junit_error": live_junit_error,
        }
    finished_at = datetime.now(UTC)
    all_items_passed = all(item["status"] == "passed" for item in acceptance_items)
    live_passed = not live_qwen or live_report["status"] == "passed"
    status = (
        "passed"
        if offline.returncode == 0 and all_items_passed and live_passed
        else "failed"
    )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": identifier,
        "status": status,
        "workspace": str(root),
        "artifact_root": str(artifact_root),
        "kept_artifacts": bool(keep_artifacts or status != "passed"),
        "live_qwen_requested": live_qwen,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "command": command,
        "return_code": offline.returncode,
        "stdout": offline.stdout,
        "stderr": offline.stderr,
        "offline_suite": {
            "status": "passed" if offline.returncode == 0 else "failed",
            "command": command,
            "return_code": offline.returncode,
            "stdout": offline.stdout,
            "stderr": offline.stderr,
            "summary": _summary(list(test_results.values())),
            "junit_error": junit_error,
        },
        "live_qwen": live_report,
        "acceptance_items": acceptance_items,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"agent_core_acceptance_{identifier}.json"
    report["report_path"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if status == "passed" and not keep_artifacts:
        artifact_root.relative_to(expected_parent)
        _remove_runner_artifacts(artifact_root)
        # Remove only empty runner-owned parents.
        for parent in (expected_parent, expected_parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--live-qwen", action="store_true")
    args = parser.parse_args(argv)
    report = run_acceptance(
        args.workspace,
        run_id=args.run_id,
        keep_artifacts=args.keep_artifacts,
        live_qwen=args.live_qwen,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

