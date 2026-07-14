# Repository Guidelines

## Project Structure & Module Organization

Production code lives in `aerospace_agent/`. The LangGraph runtime and Agent Core are under `aerospace_agent/langgraph_agent/`, while built-in aerospace tools and schemas are in `aerospace_agent/mcp/`. Domain interfaces are grouped under `aerospace_agent/domains/`; keep unimplemented experts explicitly `interface_only`. Tests mirror the package layout in `tests/`. Design specifications and execution plans live in `superpowers/specs/` and `superpowers/plans/`. Persistent project data belongs only in approved roots such as `data/langgraph/`, `memory/`, `knowledge/`, `workflows/`, and `reports/`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt` installs runtime and test dependencies.
- `python start_langgraph_agent.py --help` lists CLI options.
- `python start_langgraph_agent.py --init-project` initializes project identity and memory indexes idempotently.
- `pytest tests/langgraph_agent/agent_core -q -p no:cacheprovider --basetemp .test-artifacts/agent-core/local` runs the Agent Core suite in an isolated directory.
- `python -B -m pytest tests/langgraph_agent/test_qwen_structure_acceptance.py -m qwen3 -q -p no:cacheprovider --basetemp .test-artifacts/live-structure` runs live context, tools, Browser, Git, and TaskPlan/DAG checks against the already-running local model.
- `python -B -m pytest tests/langgraph_agent/test_qwen_long_conversation.py -m qwen3 -q -p no:cacheprovider --basetemp .test-artifacts/live-long-memory` runs the bounded 21-turn memory persistence, restart, and thread-isolation check.
- `python -B -m pytest tests/langgraph_agent/test_optical_navigation_contracts.py -q -p no:cacheprovider --basetemp .test-artifacts/optical-navigation` runs strict optical-navigation boundary and support-chain contracts; numerical estimation is intentionally not available yet.
- `python scripts/run_agent_core_acceptance.py --workspace . --live-qwen` runs mapped offline acceptance checks and the opt-in local Qwen suite.

Delete successful test artifacts. Keep failed artifacts only long enough to diagnose them; never leave `__pycache__`, `.pytest_cache`, or `pytest-cache-files-*` in the project tree.

## Coding Style & Naming Conventions

Use Python with four-space indentation and type annotations. Name modules, functions, and variables with `snake_case`; use `PascalCase` for classes and Pydantic contracts. Keep public contracts strict (`extra="forbid"`) and prefer immutable snapshots for plans, capabilities, and reviews. Use `apply_patch` for focused edits. Do not bypass `ExecutionRegistry → AuthorizedExecutor → ExecutionService` for side effects.

## Testing Guidelines

Pytest is the test framework. Name files `test_*.py` and tests `test_<behavior>`. Follow RED–GREEN–REFACTOR for behavior changes. Add negative tests for namespace isolation, path escape, confirmation replay, hash drift, checkpoint recovery, and partial-result handling. Tests requiring a running local model must use the registered `qwen3` marker or explicit live acceptance flag.

## Commit & Pull Request Guidelines

The current `.git` metadata is not usable, so no repository-specific commit convention can be evidenced. Until maintainers restore one, use concise Conventional Commit subjects such as `fix(agent-core): block expired scheduler lease`. Pull requests should state the affected contract, list exact verification commands and results, link the governing spec, and identify any unverified assumption or retained artifact.

## Evidence and Security Rules

Do not turn assumptions into conclusions. Label unsupported claims as hypotheses. Never commit secrets, approval private keys, generated databases, model outputs, or temporary fixtures. Preserve strict `(project_id, thread_id)` isolation and fail closed when evidence, approvals, hashes, or migrations are invalid.
