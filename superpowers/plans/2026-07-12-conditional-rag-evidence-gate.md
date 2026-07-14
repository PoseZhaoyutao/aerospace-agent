# Conditional RAG Evidence Gate Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ordinary conversation and work planning bypass RAG, while retaining exactly-once private-knowledge review for low-confidence domain facts, explicit evidence requests, and planner `retrieve` decisions.

**Architecture:** Add a checkpoint-safe evidence-gate decision before retrieval. The main graph routes general/confident conversation directly to synthesis, work intents to the planner, and only retrieval triggers to `rag_retrieve`; planner actions then select retrieval, tools, or response explicitly. Retrieval state resets on every new user turn and prevents repeat-query loops.

**Tech Stack:** Python 3.13, LangGraph, Pydantic 2, LangChain message types, pytest, local OpenAI-compatible Qwen endpoint.

**Approved spec:** `superpowers/specs/2026-07-12-conditional-rag-evidence-gate-design.md`

**Repository limitation:** This workspace is not a valid Git repository. Execute every commit step only if Git metadata is restored; otherwise record `not executable: no Git repository` and continue without fabricating commits.

---

## File map

- Modify `aerospace_agent/langgraph_agent/config.py`: validate the retrieval-confidence threshold.
- Modify `config/langgraph_agent.yaml`: expose the default threshold.
- Modify `aerospace_agent/langgraph_agent/state.py`: persist only primitive retrieval-decision fields.
- Modify `aerospace_agent/langgraph_agent/agent.py`: reset per-run retrieval state on follow-up turns.
- Modify `aerospace_agent/langgraph_agent/nodes.py`: implement deterministic evidence gating, conditional retrieval, explicit failure semantics, and repeated-retrieval prevention.
- Modify `aerospace_agent/langgraph_agent/graph.py`: replace mandatory RAG edges with conditional evidence/planner routing.
- Modify `tests/langgraph_agent/test_config.py`: test threshold defaults and validation.
- Modify `tests/langgraph_agent/test_schema_state.py`: test serialization/defaults of retrieval state.
- Modify `tests/langgraph_agent/test_graph_runtime.py`: test all retrieval triggers and bypass paths.
- Modify `tests/langgraph_agent/test_checkpoint_resume.py`: prove retrieval flags reset for a new turn.
- Modify `tests/langgraph_agent/test_qwen_acceptance.py`: opt-in live checks for zero-RAG chat and cited evidence requests.
- Modify `docs/LANGGRAPH_AGENT.md`: document conditional RAG behavior and metrics.

## Chunk 1: Configuration and checkpoint-safe state

### Task 1: Add a validated retrieval threshold

**Files:**
- Modify: `aerospace_agent/langgraph_agent/config.py`
- Modify: `config/langgraph_agent.yaml`
- Test: `tests/langgraph_agent/test_config.py`

- [ ] **Step 1: Write failing threshold tests**

```python
def test_retrieval_confidence_threshold_defaults_to_point_six(tmp_path):
    settings = load_settings(workspace=tmp_path)
    assert settings.knowledge.retrieval_confidence_threshold == 0.60


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_retrieval_confidence_threshold_is_probability(tmp_path, value):
    with pytest.raises(ValueError):
        AgentSettings.from_mapping(
            {"knowledge": {"retrieval_confidence_threshold": value}},
            workspace=tmp_path,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
pytest tests/langgraph_agent/test_config.py -q -p no:cacheprovider --basetemp .test-artifacts/conditional-rag-config-red
```

Expected: FAIL because `KnowledgeSettings` has no `retrieval_confidence_threshold`.

- [ ] **Step 3: Implement the minimal validated field**

```python
class KnowledgeSettings(_SettingsModel):
    workspace: Path
    data_dir: Path
    graph_output: Path
    retrieval_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
```

Add the same `0.60` value to `_default_mapping()` and `config/langgraph_agent.yaml`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Expected: all `test_config.py` tests PASS.

- [ ] **Step 5: Commit if Git is available**

```powershell
git add aerospace_agent/langgraph_agent/config.py config/langgraph_agent.yaml tests/langgraph_agent/test_config.py
git commit -m "feat: configure conditional rag threshold"
```

### Task 2: Add and reset retrieval state

**Files:**
- Modify: `aerospace_agent/langgraph_agent/state.py`
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Test: `tests/langgraph_agent/test_schema_state.py`
- Test: `tests/langgraph_agent/test_checkpoint_resume.py`

- [ ] **Step 1: Write failing serialization and follow-up reset tests**

Assert `create_initial_state()` contains:

```python
assert state["retrieval_required"] is False
assert state["retrieval_reason"] == ""
assert state["retrieval_attempted"] is False
assert state["retrieval_query_hash"] == ""
```

Add a same-thread test that seeds a completed checkpoint with retrieval fields set, calls `_input_for_run()` for a new user message, and asserts all four fields reset. This prevents a retrieval in turn N from suppressing legitimate retrieval in turn N+1.

- [ ] **Step 2: Run both focused test files and verify RED**

Expected: FAIL because the state fields and follow-up reset do not exist.

- [ ] **Step 3: Add primitive fields and per-turn reset**

Add the four fields to `AerospaceAgentState` and `create_initial_state()`. In `LangGraphAerospaceAgent._input_for_run()`, include reset values in the existing-thread delta:

```python
return {
    "messages": messages,
    "run_id": uuid.uuid4().hex,
    "retrieval_required": False,
    "retrieval_reason": "",
    "retrieval_attempted": False,
    "retrieval_query_hash": "",
    "evidence": [],
    "citations": [],
}
```

Do not store services, regex objects, or configuration models in state.

- [ ] **Step 4: Run focused state/checkpoint tests and verify GREEN**

- [ ] **Step 5: Commit if Git is available**

```powershell
git add aerospace_agent/langgraph_agent/state.py aerospace_agent/langgraph_agent/agent.py tests/langgraph_agent/test_schema_state.py tests/langgraph_agent/test_checkpoint_resume.py
git commit -m "feat: persist per-turn retrieval decisions"
```

## Chunk 2: Evidence gate and graph routing

### Task 3: Implement the pure evidence gate

**Files:**
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Test: `tests/langgraph_agent/test_graph_runtime.py`

- [ ] **Step 1: Add failing unit tests for the three trigger classes**

Use a counting knowledge double:

```python
class CountingKnowledge:
    def __init__(self, results=()):
        self.calls = []
        self.results = list(results)

    def search(self, query, *, top_k=5):
        self.calls.append((query, top_k))
        return list(self.results)
```

Test the gate directly:

- `general`, confidence `0.0`, message `你好` -> `retrieval_required=False`.
- `knowledge_query`, confidence `0.90` -> no retrieval.
- `knowledge_query`, confidence `0.40` -> reason `low_confidence`.
- message containing `请给出依据和来源` -> reason `explicit_evidence` regardless of confidence.
- English `verify this against the private knowledge base` -> `explicit_evidence`.

- [ ] **Step 2: Run the tests and verify RED**

Expected: FAIL because `evidence_gate_node` does not exist.

- [ ] **Step 3: Implement deterministic user-message and evidence-request helpers**

Add a helper that scans messages in reverse and returns the most recent human/user message. Do not use a trailing synthetic `[context]` AI message as the retrieval query.

Implement explicit evidence detection with bounded Chinese/English terms such as `依据`, `来源`, `引用`, `引文`, `审查`, `核实`, `验证`, `私域知识库`, `source`, `citation`, `evidence`, `verify`, and `audit`. Keep this a pure function.

Implement:

```python
def evidence_gate_node(state, *, confidence_threshold=0.60):
    message = _latest_user_message_text(state)
    explicit = requests_evidence(message)
    low_confidence_fact = (
        state.get("intent") == "knowledge_query"
        and float(state.get("intent_confidence", 0.0)) < confidence_threshold
    )
    reason = "explicit_evidence" if explicit else (
        "low_confidence" if low_confidence_fact else ""
    )
    return {
        "retrieval_required": bool(reason),
        "retrieval_reason": reason,
    }
```

- [ ] **Step 4: Run the gate tests and verify GREEN**

- [ ] **Step 5: Commit if Git is available**

### Task 4: Replace mandatory RAG with conditional graph edges

**Files:**
- Modify: `aerospace_agent/langgraph_agent/graph.py`
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Test: `tests/langgraph_agent/test_graph_runtime.py`

- [ ] **Step 1: Write failing graph-level bypass tests**

Add graph invocations proving:

1. General chat with an injected conversational LLM completes and `knowledge.calls == []`.
2. A confident `knowledge_query` completes with `knowledge.calls == []`.
3. A work intent reaches an injected planner before any retrieval.
4. A low-confidence factual query calls knowledge exactly once.
5. Explicit evidence wording calls knowledge exactly once and produces citations when results exist.

Use direct classifier-state tests where keyword classification cannot deterministically produce the desired confidence. Do not mock `KnowledgeService.search()` call counts indirectly through metrics.

- [ ] **Step 2: Run the new graph tests and verify RED**

Expected: ordinary chat still calls `rag_retrieve` because of the fixed edge.

- [ ] **Step 3: Add routing helpers and topology**

Add `evidence_gate` to the graph after `context_compress`.

```python
WORK_INTENTS = {
    "orbit_design", "orbit_propagation", "launch_window",
    "lunar_transfer", "maneuver_planning", "tool_discovery",
}

def _route_after_evidence_gate(state):
    if state.get("retrieval_required"):
        return "retrieve"
    if state.get("intent") in WORK_INTENTS:
        return "plan"
    return "respond"
```

Add `_route_after_planner()`:

- `retrieve` -> `rag_retrieve`, unless the same run already attempted retrieval.
- `call_tool` -> `tool_select`.
- `respond`, `stop`, protocol error, or terminal state -> `synthesize`.

Make `rag_retrieve -> planner` unconditional. Remove the intent check from `rag_retrieve_node`; conditional routing is now the gate's responsibility.

- [ ] **Step 4: Pass the configured threshold into the node**

The graph builder receives `retrieval_confidence_threshold` explicitly. `LangGraphAerospaceAgent._build_graph()` supplies `settings.knowledge.retrieval_confidence_threshold`, with `0.60` fallback for legacy callers.

- [ ] **Step 5: Run graph-runtime tests and verify GREEN**

- [ ] **Step 6: Commit if Git is available**

```powershell
git add aerospace_agent/langgraph_agent/graph.py aerospace_agent/langgraph_agent/nodes.py aerospace_agent/langgraph_agent/agent.py tests/langgraph_agent/test_graph_runtime.py
git commit -m "feat: route rag through evidence gate"
```

## Chunk 3: Exactly-once retrieval and safe degradation

### Task 5: Enforce planner-request retrieval without loops

**Files:**
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Modify: `aerospace_agent/langgraph_agent/graph.py`
- Test: `tests/langgraph_agent/test_graph_runtime.py`
- Test: `tests/langgraph_agent/test_context_cycle.py`

- [ ] **Step 1: Write failing planner retrieval tests**

Use a stateful planner double that first returns:

```python
Decision(action=ActionType.RETRIEVE, rationale="Need private evidence")
```

and after evidence returns `RESPOND`. Assert one search call and that the second planner state contains evidence.

Add a planner that always returns `RETRIEVE`; assert the graph terminates, performs one search, and emits a repeated-retrieval warning instead of hitting the recursion limit.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Mark each retrieval attempt with a normalized query hash**

In `rag_retrieve_node`, set:

```python
query = _latest_user_message_text(state)
query_hash = hashlib.sha256(" ".join(query.lower().split()).encode("utf-8")).hexdigest()
delta.update(
    retrieval_attempted=True,
    retrieval_query_hash=query_hash,
)
```

When planner action is `RETRIEVE`, set `retrieval_required=True` and reason `planner_request`. If `retrieval_attempted` already matches the current query hash, convert the repeat into a safe response/termination path with a warning.

- [ ] **Step 4: Run loop and planner tests and verify GREEN**

- [ ] **Step 5: Commit if Git is available**

### Task 6: Make retrieval failure explicit and non-fabricating

**Files:**
- Modify: `aerospace_agent/langgraph_agent/nodes.py`
- Test: `tests/langgraph_agent/test_graph_runtime.py`

- [ ] **Step 1: Write failing failure/no-hit tests**

Test explicit evidence request with:

- `search()` raising an exception;
- `search()` returning `[]`;
- malformed evidence rejected by `EvidenceItem` validation.

Expected output: `status="partial"`, no citations, a retrieval warning/error, and answer text that says verification was unavailable or asks for clarification. It must not claim that evidence supports the answer.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement minimal safe degradation**

Do not silently set `status="success"` after required retrieval fails. Preserve structured errors and make `synthesize_node` prefer a bounded verification-failure response when `retrieval_required=True` and `evidence=[]`.

- [ ] **Step 4: Run graph and schema tests and verify GREEN**

- [ ] **Step 5: Commit if Git is available**

## Chunk 4: Documentation and end-to-end verification

### Task 7: Document conditional RAG behavior

**Files:**
- Modify: `docs/LANGGRAPH_AGENT.md`
- Modify: `config/langgraph_agent.yaml`

- [ ] Document the three retrieval triggers, the default `0.60` threshold, exactly-once behavior, explicit failure semantics, and the fact that general chat does not consult private RAG by default.
- [ ] Document that `rag_hits=0` is normal for a direct conversation, not a retrieval failure.
- [ ] Check the documentation for unsupported claims, placeholders, and commands that do not match the CLI.

### Task 8: Run offline, live-Qwen, and hygiene verification

**Files:**
- Modify: `tests/langgraph_agent/test_qwen_acceptance.py`
- Generate/replace: `reports/langgraph_agent_acceptance_2026-07-12.json`
- Generate/replace: `reports/langgraph_agent_acceptance_2026-07-12.md`

- [ ] **Step 1: Add opt-in live tests**

With the already-running local endpoint:

1. Run a general chat turn and assert `citations == []` and no retrieval trigger.
2. Run an explicit `请根据私域知识库给出依据` query and assert citations contain valid page paths.

Do not start or stop the user's model process. Endpoint unavailability is `blocked/skipped`, never passed.

- [ ] **Step 2: Run focused offline tests**

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
pytest tests/langgraph_agent/test_config.py tests/langgraph_agent/test_schema_state.py tests/langgraph_agent/test_graph_runtime.py tests/langgraph_agent/test_checkpoint_resume.py -q -p no:cacheprovider --basetemp .test-artifacts/conditional-rag-focused
```

Expected: PASS with zero failures.

- [ ] **Step 3: Run all offline LangGraph tests**

```powershell
pytest tests/langgraph_agent -q -m "not qwen3" -p no:cacheprovider --basetemp .test-artifacts/conditional-rag-agent
```

- [ ] **Step 4: Run the full offline suite**

```powershell
pytest -q -m "not qwen3" -p no:cacheprovider --basetemp .test-artifacts/conditional-rag-full
```

- [ ] **Step 5: Run live Qwen acceptance**

```powershell
pytest tests/langgraph_agent/test_qwen_acceptance.py -q -m qwen3 -p no:cacheprovider --basetemp .test-artifacts/conditional-rag-qwen
```

Expected while the endpoint is available: both conditional-RAG behaviors PASS. A connection refusal is recorded as blocked.

- [ ] **Step 6: Regenerate structured acceptance evidence**

Run the existing acceptance script in an isolated workspace and ensure the report includes retrieval trigger/reason, call count, citations, and current Qwen status.

- [ ] **Step 7: Invoke `@superpowers:verification-before-completion`**

Verify the actual command outputs, not prior runs. Re-read the approved spec and map every acceptance criterion to a passing test or explicit external limitation.

- [ ] **Step 8: Clean generated test data**

After successful verification, remove `.test-artifacts/`, `.test_runs/`, `.pytest-artifacts/`, `.pytest_cache/`, and all `__pycache__` directories. Preserve formal reports and user data under `data/`.

- [ ] **Step 9: Commit if Git is available**

```powershell
git add aerospace_agent/langgraph_agent config/langgraph_agent.yaml tests/langgraph_agent docs/LANGGRAPH_AGENT.md reports/langgraph_agent_acceptance_2026-07-12.*
git commit -m "feat: make rag retrieval conditional"
```

## Final completion criteria

- [ ] General conversation and confident factual answers do not call RAG.
- [ ] Work/tool orchestration begins with the planner, not RAG.
- [ ] Low-confidence domain facts, explicit evidence requests, and planner `retrieve` decisions each call RAG exactly once.
- [ ] Repeat retrieval cannot form a graph loop.
- [ ] Failed or empty retrieval never creates citations or a false evidence-backed success.
- [ ] Retrieval fields reset per user turn and remain checkpoint-safe.
- [ ] Offline suites pass; live Qwen status is reported from a fresh run.
- [ ] All temporary test artifacts are centralized and removed after verification.
