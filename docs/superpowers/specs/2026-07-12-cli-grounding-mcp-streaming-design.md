# CLI grounding, MCP, and verified streaming design

## Scope

This change addresses three known limitations of the LangGraph aerospace
agent without expanding the seed knowledge corpus.  The command line launcher
`start_langgraph_agent.py` remains the sole supported user entry point.

## CLI boundary

All documented operations use `python start_langgraph_agent.py`.  The launcher
will expose a machine-readable `--mcp-smoke-test` action that creates the
configured stdio gateway, lists tools, calls `check_engine_availability`, and
closes the gateway.  Its JSON response records success or an explicit
`MCP_UNAVAILABLE` reason.  The launcher does not start Qwen and never silently
falls back to an in-process MCP server.

## Double grounding gate

The existing lexical and numeric provenance gate remains mandatory.  When the
configured local Qwen endpoint is available, a second judge request receives
only the candidate answer sentences and retrieved excerpts.  It must return a
strict JSON object mapping every sentence index to `supported` or
`unsupported`, with at least one evidence index for supported sentences.

The answer is accepted only when both gates accept every substantive sentence.
Malformed judge output, a request timeout, unavailable Qwen, or an unsupported
verdict leaves the lexical gate as the conservative fallback; a lexical failure
still produces the existing evidence-only answer.  Judge results and fallback
reasons are recorded as warnings/metrics, not persisted as model clients or
raw prompts in checkpoints.

## Verified streaming

`stream_text()` first executes the full graph, including retrieval, safety,
cycle detection, and double grounding.  It then yields the validated answer in
deterministic text chunks.  This is intentionally not raw model token
streaming: no unverified token is emitted.

## MCP integration evidence

The stdio gateway's existing startup/request timeout and close behavior remain
the transport control.  Unit tests keep their managed-runner skip because the
runner blocks Windows named-pipe creation.  The new CLI smoke action supplies
an executable integration check for a normal Windows workstation and returns a
structured unavailable result in restricted environments.

## Tests and acceptance criteria

1. A judge that rejects a lexically supported invented claim causes
   evidence-only fallback.
2. An unavailable or malformed judge preserves lexical-gate behavior and
   records the fallback reason.
3. Verified streaming emits more than one deterministic chunk for a long
   answer and never calls raw `stream_chat`.
4. CLI `--mcp-smoke-test` returns exactly one JSON document, closes the
   gateway on success and failure, and exposes structured unavailability.
5. Documentation lists only the CLI launcher for supported startup and smoke
   testing.
