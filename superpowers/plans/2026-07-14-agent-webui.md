# Agent WebUI Implementation Plan

> For agentic workers: use `superpowers:subagent-driven-development` or `superpowers:executing-plans` when implementing this plan. Keep each checkbox as a verifiable work item.

## Goal

Add a local-first WebUI around the existing LangGraph aerospace agent, covering chat, sessions, runtime status, errors, warnings, metrics, citations, and checkpoint-backed history without bypassing Agent Core.

## Governing documents and constraints

- Specification: `D:\Project\zytAgent\superpowers\specs\2026-07-14-agent-webui-design.md`
- Existing boundaries: `ExecutionRegistry → AuthorizedExecutor → ExecutionService`.
- Existing agent boundary: `LangGraphAerospaceAgent.run(...)` and checkpoint/history APIs.
- Phase one is local-only. No token streaming, cancellation, approval action, file browsing, Apps, Skills, Automations, tool browsing, multi-user auth, or LAN exposure.
- Do not expose secrets, raw LangGraph snapshots, arbitrary paths, Python callables, or internal services.
- The current `.git` metadata is unusable; do not claim a commit unless that is repaired and verified.

## Chunk 1: Configuration, protocol, and safe projection

### Task 1: Add strict WebUI runtime settings

**Files:**

- Modify `D:\Project\zytAgent\aerospace_agent\langgraph_agent\config.py`
- Modify `D:\Project\zytAgent\config\langgraph_agent.yaml`
- Modify `D:\Project\zytAgent\tests\langgraph_agent\test_config.py`

1. Write failing tests for a strict `webui` section with defaults `enabled=True`, `host="127.0.0.1"`, `port=8765`, `allow_lan=False`, and `auth_token_env=None`; reject invalid ports and unknown fields; compare the raw YAML `webui` mapping with `_default_mapping()["webui"]` using `yaml.safe_load()`; reject non-null `auth_token_env` with an authentication-not-implemented error.
2. Run `python -m pytest tests/langgraph_agent/test_config.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-config-red`; expect RED before implementation.
3. Implement `WebUISettings` and `AgentSettings.webui`. Preserve `allow_lan=True` as a parseable forward-compatible setting so the application can return a structured LAN-not-allowed diagnostic; the application boundary must reject it. Keep WebUI fields separate from the existing public-search `WebSettings`.
4. Run the focused test and then `python -m pytest tests/langgraph_agent/test_config.py tests/langgraph_agent/test_cli.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-config-regression`.

### Task 2: Define the versioned WebUI protocol

**Files:**

- Create `D:\Project\zytAgent\aerospace_agent\web\__init__.py`
- Create `D:\Project\zytAgent\aerospace_agent\web\protocol.py`
- Create `D:\Project\zytAgent\tests\web\__init__.py`
- Create `D:\Project\zytAgent\tests\web\test_protocol.py`

1. Write strict tests for envelope fields, `run.start`, health/thread/history/runtime models, all supported event types, empty IDs/messages, unknown fields, enum/path/AgentOutput serialization, and terminal mappings.
2. Add an explicit terminal event contract with original `status`, normalized event kind, optional bounded `reason_code`, and `reason_code="human_approval_required"` only for `run.interrupted`. Never add `human_approval_required` to `AgentOutput.status`.
3. Implement strict Pydantic v2 models and explicit unions. Do not return untyped dictionaries or raw LangGraph/service objects.
4. Run `python -m pytest tests/web/test_protocol.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-protocol`.

### Task 3: Add an allowlisted state projection

**Files:**

- Create `D:\Project\zytAgent\aerospace_agent\web\projection.py`
- Create `D:\Project\zytAgent\tests\web\test_projection.py`

1. Write tests for success, partial, interrupted, error, limit_reached, cycle_detected, model-unavailable, approval-marker precedence, and redaction.
2. Detect approval by checking `AgentOutput.status == "interrupted"` first, then bounded `errors[*].code`/`error_code` and `tool_results[*].error_code`; project it as `run.interrupted` plus `reason_code="human_approval_required"`, never as a new status.
3. Bound errors to 16 items/1024-character messages, warnings to 32 items/512 characters, metrics to depth 3 and 64 keys per object, citations to 32 items, excerpts to 4000 characters, page paths to 1024, source URIs to 2048. Redact keys matching `api_key|token|secret|password|private_key` as `[REDACTED]`.
4. Run `python -m pytest tests/web/test_projection.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-projection`.

### Chunk 1 checkpoint

- Run `python -m pytest tests/web tests/langgraph_agent/test_config.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-chunk1`.
- Remove successful artifacts and caches.

## Chunk 2: Runtime manager and gateway

### Task 4: Implement `AgentRuntimeManager`

**Files:**

- Create `D:\Project\zytAgent\aerospace_agent\web\manager.py`
- Create `D:\Project\zytAgent\tests\web\test_manager.py`
- Reference `D:\Project\zytAgent\aerospace_agent\langgraph_agent\agent.py` and `services\runtime.py`.

1. Write fake-Agent tests for one agent per workspace, server-owned thread IDs, project/workspace isolation, same-thread rejection, different-thread isolation, active disconnect recovery, bounded exception conversion, and exactly-once shutdown.
2. Assert exact mappings: success/partial → `run.completed`; interrupted → `run.interrupted`; approval marker → `run.interrupted` with `reason_code="human_approval_required"`; error/limit_reached/cycle_detected → `run.failed` preserving original status; model unavailable → structured failed event.
3. Implement dependency injection, one lock per thread, a process-local executor for blocking `agent.run()`, server-owned scope binding, and in-memory run state backed by durable history. Do not invoke tools, callables, registries, or execution services directly.
4. Run `python -m pytest tests/web/test_manager.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-manager`.

### Task 5: Add REST routes and application lifespan

**Files:**

- Create `D:\Project\zytAgent\aerospace_agent\web\routes.py`
- Create `D:\Project\zytAgent\aerospace_agent\web\app.py`
- Create `D:\Project\zytAgent\tests\web\test_routes.py`
- Modify `D:\Project\zytAgent\requirements.txt`
- Modify `D:\Project\zytAgent\setup.py`

1. Probe FastAPI/Uvicorn/websockets presence, then declare `fastapi>=0.115,<1.0` and `uvicorn[standard]>=0.30,<1.0` in both dependency files. Add `"aerospace_agent.web": ["static/**/*"]` to `setup.py` package data and retain `include_package_data=True`; Task 14 only verifies this.
2. Write failing FastAPI `TestClient` tests for health, thread creation/list/detail/history, runtime, exact 4xx errors, strict response models, raw-object exclusion, lifespan startup/shutdown, cross-project/cross-workspace rejection, and request rejection of workspace/project/path injection. Assert no route surface accepts tool catalogs, callables, or `ExecutionService`.
3. Implement the lifespan-owned manager and read-only routes plus server-side thread creation. Resolve workspace/project/factory from validated settings; do not use wildcard CORS.
4. Run `python -m pytest tests/web/test_routes.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-routes`.

### Task 6: Add the WebSocket event gateway

**Files:**

- Create `D:\Project\zytAgent\aerospace_agent\web\websocket.py`
- Create `D:\Project\zytAgent\tests\web\test_websocket.py`
- Modify `D:\Project\zytAgent\aerospace_agent\web\app.py`

1. Test connection-ready, accepted/started/completed flow, partial/interrupted/limit/cycle/error mapping, approval-marker mapping to `run.interrupted` with `reason_code="human_approval_required"`, malformed/unknown messages, cross-scope IDs, same-thread concurrency, disconnect recovery, and rejection of `run.cancel`.
2. Implement the versioned WebSocket envelope and explicit event union. Do not implement token deltas, cancellation, approvals, tool calls, or execution-service access.
3. Run `python -m pytest tests/web/test_websocket.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-websocket`.

### Chunk 2 checkpoint

- Run `python -m pytest tests/web tests/langgraph_agent/agent_core -q -p no:cacheprovider --basetemp .test-artifacts/webui-chunk2`.
- Remove successful artifacts.

## Chunk 3: Independent React WebUI

### Task 7: Scaffold the frontend

**Files:** create `D:\Project\zytAgent\webui\package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `src/main.tsx`, `src/styles/globals.css`, `src/test/setup.ts`, and `src/main.test.tsx`.

1. Verify `node --version`, `npm --version`, and `Test-Path webui\package-lock.json`; do not install global tools.
2. Create a local Vite React TypeScript app with scripts `dev`, `build`, `test`, and `typecheck`. Pin every direct dependency to an explicit version; use project-local Tailwind/shadcn prerequisites, Testing Library, and Vitest. From `webui`, run `npm install` to create/update the lockfile, then run `npm ci` as a clean install check. Record exact tool errors and mark verification unverified if unavailable.
3. Configure Vite `build.outDir = "../aerospace_agent/web/static"` and `emptyOutDir = true`; the build must produce `aerospace_agent/web/static/index.html`.
4. Add `test_main_mounts_root_element`; verify RED before implementation, then run `npm run test -- --run src/main.test.tsx`, `npm run typecheck`, and `npm run build`.

### Task 8: Implement the session shell

**Files:** create `SessionList.tsx`, `Sidebar.tsx`, `EmptyState.tsx`, `App.tsx`, and `App.test.tsx` under `D:\Project\zytAgent\webui\src`.

1. Add tests for empty state/composer, New Chat selection, session switching, local search, unsupported navigation absence, and `test_reload_restores_selected_thread_history`.
2. For the reload test, arrange a named local-storage key containing `(project_id, thread_id)` plus a mocked history response; unmount and mount a fresh `App`; assert the selected thread and assistant message restore. Repeat with mocked 404/out-of-scope response and assert identity clearing plus empty state.
3. Implement fixed sidebar, New Chat, local search, session state, reload restoration, and theme-only Settings. Persist only server-owned project/thread identity. Use mocked API functions in unit tests.
4. Run `npm run test -- --run src/app/App.test.tsx` and expect GREEN after implementation.

### Task 9: Implement transport clients

Create REST API, WebSocket, protocol decoder, and transport tests. Reject unknown backend events and malformed shapes; send versioned `run.start`; reload history on reconnect; retain thread identity after disconnect. Run the focused Vitest suite.

### Task 10: Implement chat timeline

Create `ChatWorkspace.tsx` and tests for user/assistant messages, duplicate-submit prevention while a run is active, and server-provided model/workspace labels. Do not claim token streaming.

### Task 11: Implement runtime, errors, and citations

Render running/connecting/connected/disconnected/error connection states and success, partial, interrupted, error, limit_reached, and cycle_detected terminal results. Approval is rendered as interrupted plus `reason_code="human_approval_required"`, not as a status. Never render partial/interrupted/error/limit/cycle as success. Render bounded escaped citations and collapsible warnings/metrics. Do not render approval-action, cancel, token-delta, LAN, permission, Apps, Skills, or Automations controls. Add focused Vitest tests.

### Task 12: Implement visual behavior

Use `--sidebar-width:248px`, `--content-max-width:800px`, composer radius ≥24px, and named `data-layout` regions for sidebar, empty state, composer, timeline, and runtime status. Match the supplied reference with whitespace, subtle borders/shadows, rounded composer, light/dark theme, and responsive sidebar collapse. Test preference persistence, narrow composer usability, collapse behavior, and the reference regions. Run frontend tests, typecheck, and build.

### Chunk 3 checkpoint

- Run the complete frontend unit suite from `webui`.
- Verify the Vite output path before building.
- Store screenshots/traces only under `.test-artifacts`; remove successful caches/artifacts.

## Chunk 4: Same-port serving and packaging

### Task 13: Serve static frontend and launcher

**Files:** modify `aerospace_agent/web/app.py` and `docs/LANGGRAPH_AGENT.md`; create `aerospace_agent/web/__main__.py` and `tests/web/test_static_app.py`.

1. Add tests for built index, shared API/WebSocket/static app, missing build, disabled WebUI, default localhost/8765, non-loopback rejection, `allow_lan=True` rejection without authentication, and module help that does not start an agent.
2. Use exact versioned HTTP 503 JSON diagnostics: `WEBUI_STATIC_NOT_BUILT`, `WEBUI_DISABLED`, or `WEBUI_LAN_NOT_ALLOWED`, each with a stable human message. Register API/WebSocket routes before the static fallback. Missing static assets are not silently replaced by source files.
3. Keep launcher startup separate from the task CLI, use validated settings and the existing runtime factory, and never start an LLM automatically just to print help.
4. Document backend start, Vite development proxy, production build, localhost binding, limitations, health URL, and WebSocket URL.

### Task 14: Verify package assets

**Files:** create `D:\Project\zytAgent\tests\web\test_package_assets.py`.

Test setup dependency declarations, `aerospace_agent.web` static package data, and built-distribution contents when `aerospace_agent/web/static/index.html` exists. Before Task 5, metadata tests are expected RED. If the frontend index is absent, skip only the build-dependent check with exact reason `WEBUI_FRONTEND_BUILD_UNAVAILABLE`; never call that a pass. Run the package tests and then the combined static/package suite.

### Chunk 4 checkpoint

- Run `python -m pytest tests/web/test_static_app.py tests/web/test_package_assets.py -q -p no:cacheprovider --basetemp .test-artifacts/webui-chunk4`.
- Remove successful artifacts. Commit only if Git metadata is repaired and verified; otherwise record no commit.

## Chunk 5: Acceptance and regression

### Task 15: Offline and browser acceptance

**Files:** create `tests/web/test_webui_acceptance.py` and `tests/web/test_webui_browser.py`; modify the final report only if evidence changes.

1. With a fake Agent, exercise create/run/history/citations, two-thread isolation, success/partial/interrupted, limit_reached/cycle_detected, approval reason-code mapping, model unavailable/agent errors, active disconnect/history recovery, and the invariant that non-success results are never reported as success.
2. Run `python -m pytest tests/web -q -p no:cacheprovider --basetemp .test-artifacts/webui-final` without a live Qwen endpoint.
3. Run affected existing config, CLI, checkpoint/resume, runner, and Agent Core suites.
4. If Playwright and Node are available, run the browser flow: empty state → create thread → fake-agent message → inspect history → reload → restored history. If unavailable, skip explicitly as unverified with the exact reason.
5. Inspect that no UI or route claims token streaming, cancellation, approval action, Apps, Skills, Automations, permission mode, or LAN access.

## Final verification before completion

- Every new route and WebSocket path preserves `(project_id, thread_id)` isolation.
- Manager and gateway never bypass `ExecutionRegistry → AuthorizedExecutor → ExecutionService`.
- Partial, interrupted, limit_reached, cycle_detected, error, and approval-marker results retain truthful status/event semantics.
- No secrets, generated databases, model outputs, temporary fixtures, or successful test artifacts remain in the project tree.
- All required tests pass, and any unverified optional browser/build check is explicitly reported.
