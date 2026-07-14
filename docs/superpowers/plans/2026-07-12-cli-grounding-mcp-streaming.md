# CLI grounding, MCP, and verified streaming Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CLI the sole supported entry point while adding double grounding, a real stdio MCP smoke command, and deterministic verified answer chunks.

**Architecture:** Keep `nodes.py` responsible for synthesis and provenance decisions; add a small, strict judge helper there rather than a new service.  Keep `start_langgraph_agent.py` responsible for CLI orchestration only; it owns an MCP smoke action and structured error mapping.  `stream_text()` invokes the full graph then chunks its verified answer.

**Tech Stack:** Python 3.13, LangGraph, Pydantic, local OpenAI-compatible Qwen endpoint, MCP Python SDK, pytest.

---

## Chunk 1: Double grounding

### Task 1: Strict semantic-judge tests

**Files:**
- Modify: `tests/langgraph_agent/test_graph_runtime.py`
- Modify: `aerospace_agent/langgraph_agent/nodes.py:53-82,464-520`

- [ ] **Step 1: Write failing judge tests**

```python
def test_semantic_judge_rejects_lexically_supported_answer():
    result = synthesize_node(state_with_evidence(), llm=AnswerAndRejectingJudge())
    assert "semantic grounding judge rejected" in result["warnings"][-1]
    assert result["final_answer"].startswith("Evidence:")

def test_unavailable_semantic_judge_falls_back_to_lexical_gate():
    result = synthesize_node(state_with_evidence(), llm=AnswerAndFailingJudge())
    assert "semantic judge unavailable" in result["warnings"][-1]
    assert result["final_answer"] == "The governing acceleration is -mu r / |r|^3."

def test_malformed_semantic_judge_falls_back_to_lexical_gate():
    result = synthesize_node(state_with_evidence(), llm=AnswerAndMalformedJudge())
    assert result["metrics"]["semantic_grounding"] == "malformed"
    assert result["final_answer"] == "The governing acceleration is -mu r / |r|^3."

def test_timed_out_semantic_judge_falls_back_to_lexical_gate():
    result = synthesize_node(state_with_evidence(), llm=AnswerAndTimedOutJudge())
    assert result["metrics"]["semantic_grounding"] == "timeout"
    assert result["final_answer"] == "The governing acceleration is -mu r / |r|^3."
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/langgraph_agent/test_graph_runtime.py -q`

Expected: FAIL because no semantic judge is invoked.

- [ ] **Step 3: Implement strict JSON parsing and verdict validation**

Add private helpers in `nodes.py` that: split substantive answer sentences; create a bounded judge prompt containing each indexed candidate sentence and indexed evidence excerpt; require the strict mapping `{"0":{"status":"supported","evidence_indices":[0]}}`; reject duplicate/missing indices, unknown evidence indices, non-supported states, malformed JSON, or request exceptions.  The configured local Qwen client is the only judge and is only consulted after `_answer_is_grounded()` succeeds.

- [ ] **Step 4: Connect the judge to synthesis metrics and fallback**

Record `semantic_grounding=accepted|rejected|unavailable|malformed|timeout` and a matching warning.  Reject to evidence-only rendering when the judge is available and rejects.  Preserve lexical output when the judge is unavailable, times out, or is malformed.

- [ ] **Step 4a: Assert runtime-only judge boundaries**

Add a checkpoint test that runs synthesis with a judge and verifies persisted
state/serialized checkpoint values contain neither the judge client nor the
raw judge prompt/excerpts.  Keep all prompt construction local to the node
call and only expose the enum metric and warning in state.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/langgraph_agent/test_graph_runtime.py -q`

Expected: PASS.

## Chunk 2: CLI MCP evidence and verified chunks

### Task 2: Add a structured CLI MCP smoke action

**Files:**
- Modify: `start_langgraph_agent.py:337-524`
- Modify: `tests/langgraph_agent/test_cli.py`
- Reference: `aerospace_agent/langgraph_agent/services/mcp_gateway.py`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_mcp_smoke_is_one_structured_document(tmp_path):
    result = run_cli(tmp_path, "--mcp-smoke-test")
    payload = json.loads(result.stdout)
    assert payload["status"] in {"success", "tool_unavailable"}
    if payload["status"] == "tool_unavailable":
        assert payload["error_code"] == "MCP_UNAVAILABLE"
        assert payload["reason"]
    assert "closed" in payload
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/langgraph_agent/test_cli.py::test_mcp_smoke_is_one_structured_document -q`

Expected: FAIL because the CLI option does not exist.

- [ ] **Step 3: Implement `_mcp_smoke_payload()`**

Use `create_mcp_gateway(settings.mcp)` without fallback.  List tool names, require `check_engine_availability`, call it with an empty `ToolCallRequest`, and always close in `finally`.  Return status, tool count, tool response status, and `closed`; map unavailable transport to `{status:"tool_unavailable", error_code:"MCP_UNAVAILABLE", reason:...}`.  Add injected-gateway tests proving `close()` is called after list and call failures.

- [ ] **Step 4: Add deterministic verified chunks**

Change `LangGraphAerospaceAgent.stream_text()` to execute `run()` once, then yield non-empty fixed-size chunks (for example, 256 characters).  Do not call `SimpleLLMClient.stream_chat`.  Add a test whose injected LLM raises from `stream_chat`, while a long verified answer still produces multiple concatenatable chunks.

- [ ] **Step 5: Run focused CLI and runtime tests**

Run: `pytest tests/langgraph_agent/test_cli.py tests/langgraph_agent/test_graph_runtime.py -q`

Expected: PASS; stdio tests may remain skipped in the managed runner.

## Chunk 3: Documentation and acceptance

### Task 3: Make CLI the documented boundary and validate live Qwen

**Files:**
- Modify: `docs/LANGGRAPH_AGENT.md`
- Inspect and update all user-facing `README*.md` and launcher documentation returned by `rg`/`Select-String` for prior startup examples
- Modify: `tests/langgraph_agent/test_qwen_acceptance.py` only if live semantic evidence needs an assertion
- Generate: `reports/langgraph_agent_acceptance_2026-07-12_cli-double-grounding.{md,json}`

- [ ] **Step 1: Update documentation**

Document `--mcp-smoke-test`, `--stream --task TEXT`, Qwen-as-judge behavior, fallback semantics, and the fact that only the CLI launcher is supported for user-facing starts. Search user-facing Markdown for startup commands and replace/remove conflicting entry points; record the search in the acceptance report.

- [ ] **Step 1a: Preserve no-auto-start behavior**

Add/retain a CLI test that makes endpoint probing fail and asserts a task
returns `LLM_UNAVAILABLE` without spawning a process.  The semantic judge must
reuse the already constructed Qwen HTTP client and must not add a process
launcher or subprocess dependency.

- [ ] **Step 2: Run full offline suite**

Run: `pytest tests/langgraph_agent -q -m "not qwen3"`

Expected: PASS with only managed-runner stdio skips.

- [ ] **Step 3: Run live-Qwen suite and CLI smoke commands**

Run:

```text
pytest tests/langgraph_agent -q -m qwen3
python start_langgraph_agent.py --workspace acceptance_cli_0712 --json --task "Explain the two-body governing acceleration and cite evidence."
python start_langgraph_agent.py --workspace acceptance_cli_0712 --json --mcp-smoke-test
```

Expected: Qwen tests pass; task response records semantic judge status; MCP output is either an observed success or a structured unavailable result.

- [ ] **Step 4: Regenerate acceptance report**

Run: `python scripts/run_langgraph_acceptance.py --workspace acceptance_cli_0712 --evolution-roundtrip --run-tests --json --output-json reports/langgraph_agent_acceptance_2026-07-12_cli-double-grounding.json --output-markdown reports/langgraph_agent_acceptance_2026-07-12_cli-double-grounding.md`

- [ ] **Step 5: Commit if repository ACL permits**

Run: `git add ... && git commit -m "feat: add CLI grounding and MCP smoke validation"`

Expected: commit only if `.git` write permission is available; otherwise report the ACL blocker without changing source behavior.
