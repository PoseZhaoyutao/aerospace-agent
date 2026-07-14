# Agent UX, Routing, and Memory Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make natural-language file and terminal requests execute smoothly within the confirmed safety boundary, preserve usable session memory, route work intents correctly, and enforce the requested aerospace-assistant identity.

**Architecture:** Add a small deterministic request-preparation layer beside `CapabilityRouter`; it extracts only explicit workspace paths and argv, returns validated arguments to graph state, and leaves mutations confirmation-gated. Extend terminal read-only classification, make memory extraction/search usable for Chinese and explicit conversational facts, and centralize the system prompt used by planner, classifier, and synthesis.

**Tech Stack:** Python 3.13, LangGraph, Pydantic, SQLite, pytest.

---

## Chunk 1: Natural request preparation and safe execution

**Files:**
- Modify: `aerospace_agent/langgraph_agent/agent_core/routing.py`
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Modify: `aerospace_agent/langgraph_agent/agent_core/tools/terminal.py`
- Test: `tests/langgraph_agent/agent_core/test_routing.py`
- Test: `tests/langgraph_agent/agent_core/tools/test_terminal.py`
- Test: `tests/langgraph_agent/agent_core/test_integration.py`

- [x] Write tests proving `read file AGENTS.md` produces a direct `file.read` route with validated `path`, while write/delete remain confirmation-gated.
- [x] Run the focused tests and confirm RED because natural requests currently have no parsed arguments.
- [x] Implement deterministic parsing for explicit file verbs, `file.*` names, and `terminal.run`/run-command forms; reject ambiguous or outside-workspace paths.
- [x] Persist prepared arguments and validation status in `capability_route_node` before direct execution.
- [x] Extend terminal classification so `--version`, `--help`, and equivalent read-only argv (including `python --version`) do not require mutation confirmation.
- [x] Run focused routing, integration, and terminal tests and confirm GREEN.

## Chunk 2: Durable, searchable session memory

**Files:**
- Modify: `aerospace_agent/langgraph_agent/agent_core/runtime.py`
- Modify: `aerospace_agent/langgraph_agent/agent_core/session_memory.py`
- Modify: `aerospace_agent/langgraph_agent/agent_core/context_assembler.py`
- Test: `tests/langgraph_agent/agent_core/test_session_memory.py`
- Test: `tests/langgraph_agent/test_context_cycle.py`

- [x] Add a failing integration test that writes an explicit user fact, closes the agent, reopens the same SQLite project/thread, and finds the fact in assembled context.
- [ ] Add a failing test for Chinese tokenization and accepted labels (`记住`, `约束`, `决定`, `假设`) without weakening namespace/checkpoint validation.
- [x] Implement conservative label extraction and token fallback that preserve provenance and thread isolation.
- [x] Run memory and context tests, then verify the restart test passes with SQLite.

## Chunk 3: Intent routing and assistant identity

**Files:**
- Modify: `aerospace_agent/langgraph_agent/agent_core/routing.py`
- Modify: `aerospace_agent/langgraph_agent/router.py`
- Modify: `aerospace_agent/langgraph_agent/services/planner.py`
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Test: `tests/langgraph_agent/agent_core/test_routing.py`
- Test: `tests/langgraph_agent/test_qwen_acceptance.py`

- [x] Add failing tests for orbit propagation, complex workflow planning, and ordinary conversation route selection.
- [x] Add a failing prompt-contract test asserting every model call contains the Chinese aerospace co-learning assistant identity and forbids model/vendor self-identification.
- [x] Implement deterministic work-intent fallbacks, one shared system-prompt constant, and a user-visible identity sanitizer.
- [x] Run focused route and prompt tests plus live Qwen planning and identity checks.

## Chunk 4: Full verification and cleanup

**Files:**
- Modify: `reports/agent_core_acceptance_20260714-final-full.md` only if evidence changes.

- [x] Run the focused regression suite and the full Agent Core acceptance runner with live Qwen; record the transient Windows Git-fixture caveat separately.
- [x] Verify read-only file/terminal routing, project memory status, and complex planning routes through integration coverage.
- [ ] Delete the remaining failed-run artifact directory; the automatic risk review blocked the cleanup command.
- [x] Re-scan the project tree and record exact pass/fail counts before delivery.
