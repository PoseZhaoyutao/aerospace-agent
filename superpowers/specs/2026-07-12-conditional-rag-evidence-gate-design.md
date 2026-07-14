# Conditional RAG Evidence Gate Design

**Status:** Approved in conversation on 2026-07-12; pending written-spec review

**Scope:** `aerospace_agent/langgraph_agent`, its configuration, tests, and operator documentation

## 1. Problem and verified cause

The current main graph always executes `classify_intent -> context_compress -> rag_retrieve`.
`rag_retrieve_node` also accepts both `knowledge_query` and `general`, so ordinary conversation enters the private RAG database even when no evidence is needed. This adds latency, couples every turn to the knowledge index, and contradicts the desired interaction model.

The new behavior must let the Agent converse normally and orchestrate work first. RAG is an evidence-review mechanism, not a mandatory pre-processing step.

## 2. Retrieval policy

RAG runs only when at least one of these conditions is true:

1. **Low-confidence domain fact query:** the classified intent is `knowledge_query` and `intent_confidence` is below the configurable retrieval threshold. The default threshold is `0.60`. A low-confidence `general` message does not trigger RAG.
2. **Explicit evidence request:** the user asks for a source, citation, basis, audit, verification, or private-knowledge lookup. Detection supports explicit Chinese and English evidence terms and is deterministic.
3. **Planner request:** the planner returns `Decision(action="retrieve")`.

A confident knowledge question may be answered directly unless the user requests evidence. Tool and workflow intents do not retrieve before planning merely because they mention an aerospace term.

## 3. Graph topology

The main graph becomes:

```text
START
  -> classify_intent
  -> context_compress
  -> evidence_gate
       |-- respond  -> synthesize -> END
       |-- plan     -> planner
       `-- retrieve -> rag_retrieve -> planner

planner
  |-- retrieve  -> rag_retrieve -> planner
  |-- call_tool -> tool_select -> tool_execute -> validate_output -> evaluate
  `-- respond/stop -> synthesize -> END
```

Routing rules:

- `general` conversation normally takes `respond`.
- Confident `knowledge_query` normally takes `respond`.
- Work intents such as orbit design, propagation, launch-window analysis, lunar transfer, maneuver planning, and tool discovery take `plan`.
- Any retrieval trigger takes `retrieve`.
- Retrieved evidence returns to the planner so it can answer, call a tool, or stop using the new evidence.

The legacy simple graph remains unchanged unless a failing compatibility test proves it also violates the public contract.

## 4. State and component boundaries

Add checkpoint-safe primitive channels:

- `retrieval_required: bool`
- `retrieval_reason: "" | "low_confidence" | "explicit_evidence" | "planner_request"`
- `retrieval_attempted: bool`
- `retrieval_query_hash: str`

`evidence_gate_node` decides the initial route without calling RAG. Pure routing helpers read only serialized state. `rag_retrieve_node` remains the only graph node that invokes `KnowledgeService.search()`.

Configuration adds `knowledge.retrieval_confidence_threshold`, constrained to `[0.0, 1.0]`, with default `0.60`.

No service, client, index, or callable enters checkpoint state.

## 5. Loop and failure behavior

The same normalized query may be retrieved at most once per run. If the planner requests the same retrieval again after `retrieval_attempted=True`, the graph routes to synthesis with a warning instead of looping.

RAG failure is not converted into an evidence-backed success:

- An explicit evidence request returns `partial` and states that verification failed.
- A low-confidence lookup with no usable evidence asks for clarification or gives a clearly limited answer.
- A planner-requested lookup returns control with an empty evidence set and a warning; the planner may choose a safe alternative or stop.
- Citations are emitted only for retrieved evidence actually used in the answer.

## 6. Compatibility

- `ActionType.RETRIEVE` remains the planner protocol for requesting evidence.
- Existing knowledge-query citation behavior remains available when retrieval is triggered.
- CLI commands and output schema remain compatible; metrics add retrieval decision fields without changing existing meanings.
- Existing checkpoints lacking the new optional fields receive defaults from `create_initial_state` and remain readable.

## 7. Tests and acceptance criteria

Tests must prove:

1. Ordinary conversation completes with zero `KnowledgeService.search()` calls.
2. A normal work/tool request reaches planning with zero pre-planning RAG calls.
3. A confident knowledge question can answer with zero RAG calls.
4. A low-confidence `knowledge_query` calls RAG exactly once.
5. Explicit source/citation/audit wording calls RAG exactly once regardless of intent confidence.
6. `Decision(action="retrieve")` calls RAG exactly once and returns evidence to the planner.
7. A repeated planner retrieval request does not loop.
8. RAG failure or no-hit behavior is explicit and does not fabricate citations.
9. Checkpoints contain the new primitive fields but no runtime services.
10. Existing offline LangGraph and full offline regression suites still pass.
11. When the local Qwen endpoint is available, a general conversational turn produces no RAG citations, while an explicit evidence request produces traceable citations.

All test fixtures continue to use `.test-artifacts/` and are deleted after successful verification.
