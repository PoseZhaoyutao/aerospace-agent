# Agent Loop, Browser/Web, and Multi-Model Providers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current Agent Core expose an explicit turn lifecycle, reliable session context restoration/compaction, usable public Browser/Web tools, and a configurable registry for multiple external LLM APIs.

**Architecture:** Add a small turn-orchestration layer around the existing LangGraph core. `TurnContext` and `TurnState` own restore/compact/command/build/save/respond state; `AgentRunner` owns the bounded model/tool iteration contract. Existing capability routing, conditional RAG, TaskPlan/DAG execution, review, and authorization remain inside RUN. Browser/Web providers remain read-only and workspace-bounded; SQLite remains authoritative for checkpoints and session memory.

**Tech Stack:** Python 3.10+, Pydantic v2, LangGraph, httpx, pytest.

---

## Chunk 1: Explicit AgentLoop contracts

**Files:**
- Create: `aerospace_agent/langgraph_agent/turns.py`
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Test: `tests/langgraph_agent/test_turns.py`

- [x] Write failing tests for `TurnState`, strict `TurnContext`, command shortcut, and ordered lifecycle transitions.
- [x] Run the focused tests and confirm they fail because the contracts are absent.
- [x] Implement `TurnContext`, `TurnState`, `CommandRouter`, and a lifecycle coordinator that delegates RUN to the existing graph.
- [x] Persist restore/compact/save metadata through the existing checkpoint/session services without cross-thread reads.
- [x] Run focused tests and the existing checkpoint/context tests.

## Chunk 2: AgentRunner contract

**Files:**
- Create: `aerospace_agent/langgraph_agent/runner.py`
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Test: `tests/langgraph_agent/test_runner.py`

- [x] Write failing tests for bounded iterations, tool-result feedback, cancellation, and final response normalization.
- [x] Run the tests to verify the expected failures.
- [x] Implement a provider-neutral `AgentRunner` protocol/adapter with explicit parse, execute, and feed phases; reuse the current execution boundary.
- [x] Integrate the runner into the RUN phase without bypassing `ExecutionRegistry → AuthorizedExecutor → ExecutionService`.
- [x] Run runner, routing, execution, and review tests.

## Chunk 3: Browser and public Web provider configuration

**Files:**
- Modify: `aerospace_agent/langgraph_agent/config.py`
- Modify: `config/langgraph_agent.yaml`
- Modify: `aerospace_agent/langgraph_agent/services/runtime.py`
- Modify: `aerospace_agent/langgraph_agent/agent_core/tools/web.py`
- Modify: `aerospace_agent/langgraph_agent/agent_core/tools/browser.py`
- Test: `tests/langgraph_agent/test_config.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_browser.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_web.py`

- [x] Write failing tests for named web-search providers, configured default selection, and Playwright-compatible browser screenshots.
- [x] Run the focused tests and verify failures are configuration/integration failures.
- [x] Add strict provider settings (endpoint, API key environment variable, timeout, enabled flag) and a provider chain with deterministic fallback.
- [x] Make `WebService.search` use the configured provider chain and keep public-host/credential/size guards.
- [x] Add an optional Playwright screenshot adapter factory; retain a structured `unavailable` result when Playwright is not installed.
- [x] Run all Browser/Web and security tests; no live network is required for the default suite.

## Chunk 4: Multi-model API registry

**Files:**
- Create: `aerospace_agent/langgraph_agent/providers.py`
- Modify: `aerospace_agent/langgraph_agent/config.py`
- Modify: `aerospace_agent/langgraph_agent/services/runtime.py`
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Modify: `config/langgraph_agent.yaml`
- Test: `tests/langgraph_agent/test_providers.py`
- Test: `tests/langgraph_agent/test_config.py`

- [x] Write failing tests for provider selection, environment-key resolution, OpenAI-compatible and Anthropic request mapping, and deterministic fallback.
- [x] Run the tests and confirm they fail before implementation.
- [x] Implement strict provider manifests and clients for OpenAI-compatible APIs plus a native Anthropic adapter; keep secrets out of persisted state and logs.
- [x] Make the runtime choose the configured primary model and fall back only on transport/unavailable errors.
- [x] Preserve `SimpleLLMClient` as a compatibility wrapper around the registry.
- [x] Run provider, runtime factory, prompt, routing, and live-Qwen opt-in tests.

## Chunk 5: Acceptance and cleanup

**Files:**
- Modify: `scripts/run_agent_core_acceptance.py` only if new contracts need mapping.
- Create: `reports/agent_loop-browser-model-acceptance-20260714.md`

- [x] Run focused regression tests in isolated `.test-artifacts` paths.
- [x] Run the offline acceptance suite; the opt-in live model suite was not requested and was not run.
- [x] Remove generated caches and successful test artifacts; retain only the final report and required persistent data.
- [x] Record unavailable live provider/browser conditions as explicit unverified conditions.
