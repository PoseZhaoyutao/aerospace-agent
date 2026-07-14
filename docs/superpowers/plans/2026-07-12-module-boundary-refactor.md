# LangGraph module-boundary refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the implementation conform to the approved LangGraph aerospace-agent directory and dependency boundaries before adding further behavior.

**Architecture:** Paths resolve beneath the project root; CLI `--workspace` is an explicit project-root override used for isolation/tests, and `config.py` validates every configured writable root beneath that selected project root before service construction.  Add a runtime composition root in `services/runtime.py`; it is the sole production place that builds `ContextService`, `KnowledgeService`, `EvolutionService`, a stdio `MCPGateway`, and the local LLM client.  `agent.py` accepts injected services and owns their one-time shutdown, while the CLI parses arguments, calls the composition root, and serializes results.  Graph nodes continue to depend only on `ServiceBundle` interfaces; `evolution.py` remains a compatibility delegator to `EvolutionService`.

**Tech Stack:** Python 3.13, LangGraph, Pydantic, SQLite saver, MCP SDK, pytest.

---

## Chunk 1: Composition-root boundary

### Task 1: Test and add `RuntimeServicesFactory`

**Files:**
- Create: `aerospace_agent/langgraph_agent/services/runtime.py`
- Modify: `aerospace_agent/langgraph_agent/services/__init__.py`
- Modify: `aerospace_agent/langgraph_agent/config.py`
- Modify: `tests/langgraph_agent/test_config.py`

- [ ] Write failing tests proving a factory creates `ContextService`, `KnowledgeService`, `EvolutionService`, an LLM client, and an `MCPGateway` from `AgentSettings`, and injects their runtime-only handles into the returned bundle/facade; use an injected gateway builder to avoid stdio in unit tests.
- [ ] Run the focused tests and confirm failure because no composition root exists.
- [ ] Implement `RuntimeServicesFactory.create()` with typed runtime-only dependencies.  It must construct `EvolutionService` too, call `create_mcp_gateway()` for production, select stdio under production settings, preserve explicit degraded-mode warnings, and never put clients in graph state.  The Agent, not the factory, owns closing constructed services.
- [ ] Run the focused tests and confirm pass.

## Chunk 2: Agent and CLI dependency direction

### Task 2: Remove direct registry assembly from the public startup path

**Files:**
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Modify: `start_langgraph_agent.py`
- Modify: `tests/langgraph_agent/test_cli.py`
- Modify: `tests/langgraph_agent/test_graph_runtime.py`

- [ ] Write failing tests asserting CLI-created agents receive non-null context and an object implementing `MCPGateway`, not a tool dictionary.
- [ ] Run focused tests and confirm failure against the current `_safe_tools()` path.
- [ ] Modify the factory/launcher integration so `_create_agent()` creates runtime services through the composition root; pass only the resulting `ServiceBundle` to `LangGraphAerospaceAgent`.  Keep legacy direct injection only as an explicit compatibility/test path, not CLI behavior.
- [ ] Ensure `agent.close()` closes the checkpointer and runtime gateway exactly once; test idempotent repeated close without double-closing either resource.
- [ ] Run focused tests and confirm pass.

### Task 3: Repair the CLI stream contract

**Files:**
- Modify: `start_langgraph_agent.py`
- Modify: `tests/langgraph_agent/test_cli.py`

- [ ] Write a failing test for `--stream --task TEXT` returning one JSON document with chunks.
- [ ] Replace the mutually-exclusive boolean `--stream` contract with a stream task action that can receive text without conflicting with `--task`.
- [ ] Run the focused test and confirm pass.

## Chunk 3: Required workspace directories and boundary tests

### Task 4: Materialize and protect declared runtime directories

**Files:**
- Modify: `aerospace_agent/langgraph_agent/services/runtime.py`
- Modify: `aerospace_agent/langgraph_agent/services/knowledge.py` only if initialization needs a narrow hook
- Modify: `aerospace_agent/langgraph_agent/services/evolution.py`
- Create: `evolved_skills/.gitkeep`
- Create: `workflows/evolved/.gitkeep`
- Modify: `tests/langgraph_agent/test_knowledge_service.py`

- [ ] Write a failing test that factory initialization creates `knowledge`, `memory`, `evolved_skills`, `workflows/evolved`, and `data/langgraph/{rag,artifacts,evolution}` beneath the selected workspace.
- [ ] Add failing configuration tests that reject `..` and symlink/junction escapes for every configured writable root before directory materialization.
- [ ] Add failing `EvolutionService` tests that reject proposal targets escaping through `..` or a symlink/junction, and reject all proposal roots other than `knowledge/`, `memory/`, `evolved_skills/`, and `workflows/evolved/`; transaction-owned staging, backup, manifests, and reports remain internal under `data/langgraph/evolution/<id>/`.
- [ ] Implement idempotent directory materialization in the composition root; resolved paths must stay below the selected workspace and evolution roots.
- [ ] Implement proposal-target validation in `services/evolution.py` by resolving each target against its allowed root and rejecting `..`, symlink/junction traversal, and roots outside the proposal whitelist before staging any file.
- [ ] Run the focused test and confirm pass.

### Task 5: Enforce dependency direction

**Files:**
- Create: `tests/langgraph_agent/test_module_boundaries.py`
- Modify: `docs/LANGGRAPH_AGENT.md`

- [ ] Write AST/text boundary tests: `nodes.py` may use `ServiceBundle` but may not import or instantiate concrete context, knowledge, MCP, LLM, filesystem/index, or evolution implementations; `start_langgraph_agent.py` and `agent.py` may not construct `ContextService`, `KnowledgeService`, `EvolutionService`, an MCP implementation, or an LLM client; `services/runtime.py` is the only production caller of `create_mcp_gateway` and concrete service constructors.
- [ ] Add a checkpointer serialization test proving `ServiceBundle`, MCP gateway, LLM client, and factory objects are absent from persisted state.
- [ ] Add a production-vs-degraded MCP test: production settings request stdio; in-process use is accepted only through an explicit test/degraded path and produces a warning.
- [ ] Add an opt-in real stdio protocol smoke test using the official MCP client: initialize, list tools, verify definitions match the handler registry, call `check_engine_availability`, exercise invalid arguments and an unavailable engine structured error, close, and assert server stdout has no non-protocol startup text.  It may skip only when the managed Windows runner blocks named pipes.
- [ ] Run tests to verify current violations are detected before changing production paths.
- [ ] Implement only the changes needed to make the dependency tests pass; document the composition-root boundary and CLI-only launch path.
- [ ] Run boundary and documentation tests.

## Chunk 4: Regression and handoff

### Task 6: Preserve the evolution compatibility boundary

**Files:**
- Modify: `aerospace_agent/langgraph_agent/evolution.py`
- Modify: `tests/langgraph_agent/test_evolution_service.py`

- [ ] Write a failing test proving legacy imports from `evolution.py` delegate to `services.evolution.EvolutionService` and do not define file-system transaction helpers.
- [ ] Reduce `evolution.py` to documented compatibility exports/adapters only; keep transaction staging, validation, rebuild, commit, and rollback in `services/evolution.py`.
- [ ] Run evolution tests and confirm pass.

### Task 7: Verify all Agent behavior after refactor

**Files:**
- Generate: `reports/langgraph_agent_module_boundary_2026-07-12.{md,json}`

- [ ] Run `pytest tests/langgraph_agent -q -m "not qwen3"`.
- [ ] Run `pytest tests/langgraph_agent -q -m qwen3` against the running local endpoint.
- [ ] Run CLI knowledge, task, stream, checkpoint, and MCP-unavailable paths in isolated workspaces; capture only structured evidence.
- [ ] Regenerate a focused acceptance report and disclose any managed-runner stdio restriction.
