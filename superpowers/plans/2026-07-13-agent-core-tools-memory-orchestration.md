# Agent Core Tools, Memory, and Orchestration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Agent Core design as self-contained, testable runtime capabilities with strict project/thread isolation, conditional RAG, safe execution, durable planning, and explicit unavailable states.

**Architecture:** Add a focused `agent_core` package underneath the existing LangGraph agent. Contracts and persistence are independent of graph nodes; the graph and CLI consume the services through one `ExecutionService`. Existing aerospace MCP tools remain compatible, but accidental imports from adjacent repositories are removed and every advertised capability must resolve to an in-repository executor.

**Tech Stack:** Python 3.13, Pydantic v2, SQLite/FTS5, LangGraph, PyYAML, httpx, cryptography Ed25519, pytest.

**Authority:** `superpowers/specs/2026-07-13-agent-core-tools-memory-orchestration-design.md`.

**Repository constraint:** `git rev-parse --show-toplevel` currently fails. Commit/worktree steps are therefore recorded as unavailable and must not be reported as completed. No destructive Git operation is authorized.

**Test-data constraint:** Set `AEROSPACE_TEST_ARTIFACT_ROOT=.test-artifacts/agent-core/<run-id>`. The repository overrides `tmp_path`; `--basetemp` alone does not contain test output. No test writes to user `data/`, `memory/`, `workflows/`, or the project root.

---

## Chunk 1: Core Contracts, Registry, Confirmation, and Routing

### Task 1: Define immutable public contracts

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/__init__.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/models.py`
- Test: `tests/langgraph_agent/agent_core/test_models.py`

- [ ] Write failing tests for strict Pydantic validation and canonical SHA-256 identity of `CapabilityManifest`, `ToolCall`, `ToolResult`, `ToolError`, `ConfirmationGrant`, `OperationRecord`, `TaskPlan`, `PlanExecutionState`, `ReviewResult`, `ExecutionRun`, `WorkflowManifest`, and domain artifact/handoff contracts.
- [ ] Verify RED with `pytest tests/langgraph_agent/agent_core/test_models.py -q`.
- [ ] Implement the exact fields/enums from design §§4–11 with `extra="forbid"`, UTC timestamps, and canonical JSON hashing.
- [ ] Verify GREEN and reject unknown error codes, mutable plan status embedded in `TaskPlan`, and executor variants missing their required discriminator-specific identifier.

### Task 2: Create SQLite schema and atomic primitives

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/store.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/migrations/session_memory/001_initial.sql`
- Create: `aerospace_agent/langgraph_agent/agent_core/migrations/project_index/001_initial.sql`
- Create: `aerospace_agent/langgraph_agent/agent_core/migrations/execution/001_initial.sql`
- Create: `aerospace_agent/langgraph_agent/agent_core/migrations/scheduler/001_initial.sql`
- Create: `aerospace_agent/langgraph_agent/agent_core/migrations/approval/001_initial.sql`
- Test: `tests/langgraph_agent/agent_core/test_store.py`

- [ ] Write failing tests for independent versioned databases, sequential `PRAGMA user_version` migration, schema idempotence, WAL/foreign-key setup, rollback on migration failure, service-local disablement, legacy-project migration, atomic confirmation consume, operation state transitions, RAG claim/lease recovery, scheduler lease claiming, and strict `(project_id, thread_id)` filtering.
- [ ] Verify RED.
- [ ] Implement one transaction boundary per state transition using `BEGIN IMMEDIATE`; never implement check-then-write in separate transactions. A migration failure disables only its owning service and does not corrupt or disable unrelated stores.
- [ ] Verify GREEN including two-connection contention tests.

### Task 3: Make capability registration self-contained

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/capabilities.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/execution.py`
- Modify: `aerospace_agent/mcp/tools/__init__.py`
- Modify: `aerospace_agent/langgraph_agent/services/mcp_gateway.py`
- Test: `tests/langgraph_agent/agent_core/test_capabilities.py`
- Test: `tests/langgraph_agent/test_module_boundaries.py`

- [ ] Write a failing test proving tool discovery never imports `aerospace_agent.research_tools` from another checkout and every `available` manifest resolves to exactly one `AuthorizedExecutor` for all five executor kinds.
- [ ] Reject raw callable leakage and validate allowed import roots, current plan snapshot, entrypoint/dependency/cache hashes, input Schema, confirmation binding, and human-step interruption before authorization.
- [ ] Verify RED against the current lazy `_register_research_tools` behavior.
- [ ] Implement `CapabilityRegistry`, `ExecutionRegistry.resolve(...) -> AuthorizedExecutor`, and the sole `ExecutionService.execute(authorized_executor)` path. No public API returns the underlying callable.
- [ ] Delete the implicit research-tool registration path; optional integrations must be registered only by an explicit current-project manifest.
- [ ] Verify GREEN and assert duplicate executor names fail closed.

### Task 4: Implement capability routing and confirmation boundary

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/routing.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/confirmation.py`
- Modify: `aerospace_agent/langgraph_agent/services/planner.py`
- Modify: `aerospace_agent/langgraph_agent/router.py`
- Test: `tests/langgraph_agent/agent_core/test_routing.py`
- Test: `tests/langgraph_agent/agent_core/test_confirmation.py`

- [ ] Write failing `CapabilityRoute` contract and audit tests for the exact outcomes `conversation`, `knowledge_qa`, `direct_execution`, `complex_task`, `memory_operation`, `project_operation`, and `clarify`; cap candidates at 12 and record the complete routing audit. Include negative RAG requests, low confidence, evidence requests, and planner-requested retrieval.
- [ ] Write failing confirmation tests binding project/thread/root-run/operation/action hash, expiry, and atomic single use.
- [ ] Verify RED.
- [ ] Implement deterministic policy gates before any LLM hint and persist confirmation grants before use.
- [ ] Verify GREEN; the model may propose but cannot bypass policy.

## Chunk 2: Project Identity, Session Memory, and Conditional RAG

### Task 5: Initialize project identity and global memory

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/project.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/memory/project_memory.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/memory/templates/PROJECT.md`
- Create: `aerospace_agent/langgraph_agent/agent_core/memory/templates/constraints.yaml`
- Create: `aerospace_agent/langgraph_agent/agent_core/memory/templates/manifest.yaml`
- Modify: `start_langgraph_agent.py`
- Test: `tests/langgraph_agent/agent_core/test_project_init.py`
- Test: `tests/langgraph_agent/test_cli.py`

- [ ] Write failing tests for `--init-project`, stable project ID, `.init.lock`, idempotent rerun, partial-init recovery, and `project_not_initialized`/migration errors.
- [ ] Add CLI tests for `--project-memory-status` returning the exact three lifecycle states and for `--project-memory-reindex` rebuilding only the project index.
- [ ] Verify RED.
- [ ] Implement atomic staging/replace within the workspace; create only the design-mandated memory/database structure.
- [ ] Verify GREEN, including legacy CLI behavior before initialization.

### Task 6: Implement durable isolated session memory

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/memory/session_memory.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/memory/context_assembler.py`
- Modify: `aerospace_agent/langgraph_agent/services/context.py`
- Modify: `aerospace_agent/langgraph_agent/graph.py`
- Test: `tests/langgraph_agent/agent_core/test_session_memory.py`
- Test: `tests/langgraph_agent/agent_core/test_context_assembler.py`

- [ ] Write failing tests proving same-thread facts survive process restart, different thread/project facts never appear, checkpoint refs must share namespace, and FTS results are joined through both IDs.
- [ ] Cover every `SessionMemory` field/truth state, extraction only after checkpoint success, user correction via `supersedes`, default exclusion of superseded/retracted entries, and provenance requirements.
- [ ] Test `memory.remember/search/list/update/forget/clear`; callers may never supply arbitrary project/thread IDs, and forget/clear require confirmation at the specified scope.
- [ ] Verify RED; record that the current role-count summary loses facts.
- [ ] Implement append-only turns plus explicit summaries/facts/artifact/checkpoint refs; assemble context from bounded recent turns and retrieved session facts.
- [ ] Verify GREEN without weakening the existing context token bound.

### Task 7: Enforce one retrieval budget per root run

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/retrieval.py`
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Modify: `aerospace_agent/langgraph_agent/state.py`
- Test: `tests/langgraph_agent/agent_core/test_retrieval_budget.py`
- Test: `tests/langgraph_agent/test_graph_runtime.py`

- [ ] Write failing tests for `available → claimed → in_flight → consumed`, crash-before-request release, crash-after-request `consumed_unknown`, confirmation continuation retaining root run, and scheduled run budget zero.
- [ ] Verify RED.
- [ ] Implement atomic claims and fail closed when retrieval outcome is indeterminate.
- [ ] Verify GREEN and preserve the three allowed RAG triggers only.

## Chunk 3: Basic Tools and Safe Execution

### Task 8: Implement file and terminal tools

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/tools/__init__.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/tools/files.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/tools/terminal.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/journal.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_files.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_terminal.py`

- [ ] Write failing tests for read/list/stat/search, atomic write/append/mkdir/move/copy/delete, symlink escape, preimage backup, command allowlisting, timeout, cancellation, output caps, and side-effect classification.
- [ ] Verify RED.
- [ ] Implement project-root path normalization and operation journaling. Only FileService writes with verified preimages may claim reversible rollback; terminal writes default to manual recovery.
- [ ] Verify GREEN.

### Task 9: Implement web search and public read-only browser

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/tools/web.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/tools/browser.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_web.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_browser.py`

- [ ] Write failing tests using injected transports for search, fetch, confirmation-gated download with expected-hash verification, URL validation, redirect/private-network blocking, read-only page open/follow-link/extract/screenshot, and disabled login/form/upload/payment actions.
- [ ] Verify RED.
- [ ] Implement `httpx` search/fetch/download adapters with explicit provider configuration. Add an optional Playwright screenshot adapter; keep only that operation `unavailable` while Playwright/browser binaries are absent, and never mark the browser capability available solely because a manifest exists.
- [ ] Verify GREEN without network access; live tests remain opt-in.

### Task 10: Register memory, scheduler, workflow, Git, and capability tools

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/tools/system.py`
- Modify: `aerospace_agent/mcp/tools/__init__.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_tool_catalog.py`

- [ ] Write a failing catalog test for at least 30 self-contained basic tools across file, terminal, browser, web, schedule, memory, Git, workflow, capability, and SpaceBasic groups, with exact manifest/executor parity.
- [ ] Verify RED.
- [ ] Register only services already built and validated. Future Scheduler/Workflow/Git entries remain manifest status `unavailable` until their executor, migrations, and contract tests pass; Git remains `unavailable` in the current invalid repository instead of claiming success.
- [ ] Verify GREEN and export JSON schemas through the existing CLI.

## Chunk 4: TaskPlan, DAG Execution, Review, and Aerospace Interfaces

### Task 11: Implement immutable plans and checkpointed DAG execution

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/planning.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/dag.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/review.py`
- Test: `tests/langgraph_agent/agent_core/test_planning.py`
- Test: `tests/langgraph_agent/agent_core/test_dag.py`
- Test: `tests/langgraph_agent/agent_core/test_review.py`

- [ ] Write failing tests for canonical `plan_sha256`, revision/supersedes, snapshot mismatch block, dependency cycle rejection, checkpoint resume, and the `PlanStep` discriminated-union table. Reject unavailable capabilities, high-risk steps without confirmation, non-canonical units/frame/time metadata, missing cross-domain conversion/verification, and every `invalid_plan` without checkpoints or execution side effects.
- [ ] Add DAG tests for idempotency key `(plan_id, step_id, input_hash)`, checkpoints before/after every step, and interrupted write-state inspection before retry.
- [ ] Add `ReviewResult` tests for identity, evidence sufficiency, tool safety, domain review, boundary review, and the rule that partial/failed work cannot be declared complete.
- [ ] Verify RED.
- [ ] Implement immutable plan storage separated from mutable execution state; use only `ExecutionService` for steps.
- [ ] Verify GREEN. Completion requires a passing review tied to exact plan/root-run/project/thread IDs.

### Task 12: Add ArtifactStore and six interface-only domain subgraphs

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/artifacts.py`
- Create: `aerospace_agent/domains/__init__.py`
- Create: `aerospace_agent/domains/base.py`
- Create: `aerospace_agent/domains/simulation/__init__.py`
- Create: `aerospace_agent/domains/navigation_orbit_determination/__init__.py`
- Create: `aerospace_agent/domains/control_planning/__init__.py`
- Create: `aerospace_agent/domains/orbit_design/__init__.py`
- Create: `aerospace_agent/domains/mechanical_thermal_electrical/__init__.py`
- Create: `aerospace_agent/domains/fault_diagnosis_maintenance/__init__.py`
- Test: `tests/langgraph_agent/agent_core/test_artifacts.py`
- Test: `tests/langgraph_agent/test_domain_interfaces.py`

- [ ] Write failing tests that make `ArtifactStore` the only resolver; require project-local URI, SHA-256, byte length, approved `ArtifactSchemaManifest`, actual metadata, source snapshot, and same-project/thread checkpoint identity.
- [ ] Require concrete `HandoffRecord` source artifact IDs plus validation of source payload, conversion output, target payload, and every metadata conversion.
- [ ] Reject execution by all six domain placeholders by returning a `CapabilityGap` only; assert no PlanStep, checkpoint, or artifact is created.
- [ ] Verify RED.
- [ ] Implement `ArtifactStore` and interface manifests with status `interface_only`; do not implement or advertise domain algorithms.
- [ ] Verify GREEN.

### Task 13: Gate SpaceBasicTools by real adapters

**Files:**
- Modify: `aerospace_agent/mcp/tools/space_tools.py`
- Modify: `aerospace_agent/mcp/tools/__init__.py`
- Test: `tests/test_space_tool_specs.py`

- [ ] Write failing tests for the exact seven `space.*` names and for unavailable status when dependency/adapter/contract tests are missing. Validate each manifest's input/output schema, units, frame, time system, assumptions, risk, validator, dependency version, and adapter hash.
- [ ] Verify RED.
- [ ] Map existing local implementations only where their adapter availability and contract checks pass; otherwise keep unavailable.
- [ ] Verify GREEN; capability count is not evidence of executable availability.

## Chunk 5: Workflows, Scheduler, Git/Recovery, and Evolution

### Task 14: Implement immutable approved workflows

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/workflows.py`
- Create: `schemas/workflow_manifest.schema.json`
- Test: `tests/langgraph_agent/agent_core/test_workflows.py`

- [ ] Write failing tests for canonical workflow/version hashes, exact input/payload schema, sensitive-field masking, step policy completeness, input validation, approval binding, and auto-run eligibility.
- [ ] Verify RED.
- [ ] Implement registry/load/validate/invoke through `ExecutionService`.
- [ ] Verify GREEN; only all-read-only, idempotent, approved workflows may auto-run.

### Task 15: Implement internal SQLite scheduler

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/scheduler.py`
- Modify: `start_langgraph_agent.py`
- Test: `tests/langgraph_agent/agent_core/test_scheduler.py`
- Test: `tests/langgraph_agent/test_cli.py`

- [ ] Write failing tests for content-addressed payloads, immutable workflow locks, atomic lease/cancel/retry, overdue recovery, attempt root-run IDs, and no-RAG scheduled execution.
- [ ] On every claim, require revalidation of payload schema/hash, workflow body/manifest hashes, approval record, `scheduled_read_only`, `automatable`, and every step's `read_only`, `recovery_class=read_only`, and idempotency flags.
- [ ] Cover the full job state machine, optimistic `(job_id, version)` cancellation, `cancel_requested` safe-point checks, non-interruptible external work becoming `manual_recovery`, and the exact allowlist for retry and overdue recovery.
- [ ] Verify RED.
- [ ] Implement queue worker and CLI create/list/cancel/run-due actions. Do not integrate Windows Task Scheduler.
- [ ] Verify GREEN with deterministic clock injection.

### Task 16: Implement Git detection and truthful recovery

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/git_service.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/recovery.py`
- Test: `tests/langgraph_agent/agent_core/test_git_service.py`
- Test: `tests/langgraph_agent/agent_core/test_recovery.py`

- [ ] Write failing tests for repository detection, read-only status/diff/log, scoped commit confirmation, complete recovery states, pre/post hashes, backup/verification records, file preimage rollback, compensatable recovery, and manual-recovery reports.
- [ ] Assert `reset --hard`, force push, and branch deletion are never offered by the service, and only explicitly eligible operation classes may claim `reversible`.
- [ ] Verify RED.
- [ ] Implement non-interactive subprocess calls with project scope and confirmation. Never represent terminal/process changes as universally reversible.
- [ ] Verify GREEN using a fixture repository, while the current project correctly reports Git unavailable.

### Task 17: Implement evidence-bound evolution and capability acquisition

**Files:**
- Create: `aerospace_agent/langgraph_agent/agent_core/evolution.py`
- Create: `aerospace_agent/langgraph_agent/agent_core/integrations.py`
- Modify: `aerospace_agent/langgraph_agent/services/evolution.py`
- Test: `tests/langgraph_agent/agent_core/test_evolution.py`
- Test: `tests/langgraph_agent/agent_core/test_integrations.py`

- [ ] Write failing tests for human-approved session-to-project promotion, workflow candidate staging, and Ed25519 verification using a workspace-external private key plus runtime-only public key/key ID over the specified combined digest.
- [ ] Test all three allowed import roots, lock files, content-addressed read-only Git cache, `local_code` boundaries, source/manifest/adapter hash drift forcing `unavailable` and reapproval, license/version evidence, one build attempt per capability/run, and resume from the original plan checkpoint.
- [ ] Verify RED.
- [ ] Implement candidate generation and approval ledger. Each integration trust anchor lives at `aerospace_agent/integrations/<capability_id>/manifest.yaml`; signing authority remains outside the workspace. Searching is read-only; download/install/clone/execute remains confirmation-gated.
- [ ] Verify GREEN; no unapproved candidate enters the active registry.

## Chunk 6: Integration, CLI Acceptance, and Data Hygiene

### Task 18: Wire services into the agent and CLI

**Files:**
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Modify: `aerospace_agent/langgraph_agent/graph.py`
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Modify: `aerospace_agent/langgraph_agent/schema.py`
- Modify: `aerospace_agent/langgraph_agent/state.py`
- Modify: `aerospace_agent/langgraph_agent/config.py`
- Modify: `config/langgraph_agent.yaml`
- Modify: `start_langgraph_agent.py`
- Test: `tests/langgraph_agent/agent_core/test_integration.py`
- Test: `tests/langgraph_agent/test_qwen_acceptance.py`

- [ ] Write failing end-to-end tests for all seven route outcomes in design §3.1, same-thread restart recall, cross-thread isolation, simple tool direct execution, complex-plan checkpoint/review, conditional RAG, and unavailable domain/tool handling.
- [ ] Verify RED.
- [ ] Wire the Agent Core services through explicit dependency injection and retain legacy behavior for uninitialized projects.
- [ ] Verify GREEN with deterministic doubles, then run the opt-in live Qwen suite against the already-running local model.

### Task 19: Acceptance runner and cleanup

**Files:**
- Create: `scripts/run_agent_core_acceptance.py`
- Create: `reports/agent_core_acceptance_2026-07-13.md`
- Modify: `.gitignore`
- Test: `tests/langgraph_agent/agent_core/test_acceptance_runner.py`

- [ ] Write failing tests that the runner uses a unique `.test-artifacts/agent-core/<run-id>`, writes one machine-readable report, and cleans the run directory only on success unless `--keep-artifacts` is passed.
- [ ] Verify RED.
- [ ] Implement runner and report generation.
- [ ] Bind the runner report one-to-one to every acceptance item in design §13, including 30+ tool inventory, strict namespace isolation, conditional RAG, scheduler, workflow snapshots, recovery truthfulness, domain interface-only behavior, and signed integration controls.
- [ ] Run focused tests, full offline tests, live Qwen tests, and every §13 acceptance check.
- [ ] Inspect all allowed persistence roots plus the project tree for stray `.tmp`, `.test-artifacts`, `__pycache__`, pytest cache, generated databases, fixtures, and stale reports. Remove only artifacts created by this implementation; preserve all pre-existing user data.
- [ ] Assert that surviving generated files exist only in the design-approved persistence/report directories.
- [ ] Run the LoopRecursive CEO `FINALIZATION_ATTEMPT`; report incomplete if any required acceptance criterion lacks current evidence.
