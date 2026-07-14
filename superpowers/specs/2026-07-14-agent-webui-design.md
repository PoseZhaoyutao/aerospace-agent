# Agent WebUI Design Specification

## 1. Scope

Add a local-first browser WebUI around the existing LangGraph aerospace agent. Phase one covers:

- chat messages and server-backed sessions;
- runtime connection/run status;
- bounded errors, warnings, metrics, citations, and checkpoint metadata;
- strict `(project_id, thread_id)` isolation;
- a same-process FastAPI/REST/WebSocket gateway and a React/TypeScript/Vite frontend.

Phase one does not implement token streaming, cancellation, approval actions, file browsing, MCP tool browsing, Apps, Skills, Automations, multi-user authentication, LAN exposure, or a new execution path.

## 2. Reference direction

Use the provided reference image as the visual direction: fixed left rail, New Chat, local search, generous empty state, rounded composer, subtle borders/shadows, theme toggle, and responsive collapse. Unsupported navigation items remain hidden.

Use nanobot WebUI only as an architectural reference: independent React/TypeScript source, Vite build, WebSocket multiplexing, and same-port REST/static serving. Do not copy nanobot protocol or security assumptions.

## 3. Backend architecture

```text
Browser
  ├─ REST: health, runtime, threads, history
  └─ WebSocket: versioned run envelopes and terminal events
        │
FastAPI Web Gateway
        │
AgentRuntimeManager
        │
LangGraphAerospaceAgent.run(...)
        │
ExecutionRegistry → AuthorizedExecutor → ExecutionService
```

The gateway and manager may call the existing agent boundary only. They must not expose tools, Python callables, raw LangGraph snapshots, arbitrary paths, or internal execution services.

The manager owns one validated workspace/project scope, one agent instance per workspace, one active run per thread, and bounded in-memory connection state. Durable history/checkpoints remain the source for recovery after a browser disconnect.

## 4. Protocol

All REST and WebSocket payloads are strict Pydantic v2 models with `schema_version="1.0.0"` where applicable. WebSocket events include `request_id` and `thread_id`.

Supported events: `connection.ready`, `run.accepted`, `run.started`, `run.completed`, `run.interrupted`, and `run.failed`.

Terminal events preserve the original `AgentOutput.status`. Normalized event mapping is:

| Agent result | WebSocket event | Preserved status |
|---|---|---|
| success/partial | `run.completed` | original value |
| interrupted | `run.interrupted` | `interrupted` |
| approval marker | `run.interrupted` | original interrupted status |
| error/limit_reached/cycle_detected | `run.failed` | original value |

Approval is represented only as `reason_code="human_approval_required"` on `run.interrupted`. It is derived from allowlisted existing fields (`AgentOutput.status`, bounded error codes, and bounded tool-result error codes); do not invent a new `AgentOutput.status` value.

## 5. REST and WebSocket surface

REST:

- `GET /api/v1/health`
- `GET /api/v1/runtime`
- `GET /api/v1/threads`
- `POST /api/v1/threads`
- `GET /api/v1/threads/{thread_id}`
- `GET /api/v1/threads/{thread_id}/history`

WebSocket:

- `GET /api/v1/ws`
- client envelope `run.start` with non-empty message;
- unsupported `run.cancel`, token-delta, approval-action, and tool-control messages are rejected as structured unsupported operations.

The root path serves the built frontend from `aerospace_agent/web/static/index.html` after the frontend build. API/WebSocket routes are registered before the static fallback.

## 6. Security and configuration

Configuration adds a strict `webui` section:

```yaml
webui:
  enabled: true
  host: 127.0.0.1
  port: 8765
  allow_lan: false
  auth_token_env: null
```

`allow_lan` is retained as a forward-compatible field but phase one rejects `true` at the application boundary because authentication is not implemented. Any non-loopback bind is rejected. Non-null `auth_token_env` is rejected. No wildcard CORS is added.

Every request is resolved against server-owned workspace/project settings; request bodies cannot choose workspace, project, or filesystem paths. Error, warning, metric, citation, and excerpt fields are bounded and secret-looking keys are redacted.

## 7. Frontend

The independent `webui/` app uses React 18, TypeScript, Vite, Tailwind, shadcn/ui prerequisites, Testing Library, and Vitest. The Vite output is `aerospace_agent/web/static`.

The shell contains the fixed sidebar, New Chat, local session search, session list, empty state, composer, message timeline, runtime status, error cards, citation cards, and theme toggle. It does not claim token streaming or render unsupported controls.

## 8. Verification gates

Tests must cover configuration, protocol strictness, projection bounds/redaction, manager scope/concurrency/recovery, REST isolation, WebSocket event mapping, frontend shell/transport/status behavior, static serving, package assets, and offline end-to-end acceptance with a fake Agent. Existing Agent Core suites must remain green. Browser acceptance is optional only when its exact dependency is unavailable and must be reported as unverified.
