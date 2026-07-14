# LangGraph Aerospace Agent

This document describes the supported local runtime and its command-line
boundary.  The CLI is deliberately local-first: it reads one validated
settings object, keeps runtime clients out of checkpoints, and writes all
artifacts below the selected workspace.

## Architecture boundaries

`start_langgraph_agent.py` is an adapter, not a second agent implementation.
It loads `AgentSettings`, constructs a `LangGraphAerospaceAgent`, and converts
the typed `AgentOutput` to JSON.  The graph owns planning, bounded loops,
retrieval, tool calls, synthesis, and output validation.  `KnowledgeService`
owns the Markdown Wiki and rebuilds its derived RAG index and graph.  The MCP
gateway is the only boundary for registered tools; tool requests and
responses are validated with Pydantic schemas.  Evolution is a separate,
journaled transaction service and is never allowed to modify source code
outside its configured roots.

Only serializable state channels are persisted in LangGraph checkpoints.
Network clients, database handles, and indexes are runtime dependencies and
must not be placed in graph state.

## Conditional private RAG

Private RAG is an evidence-review path, not a mandatory step for every turn.
General conversation and confident knowledge answers bypass the private index.
Work and tool intents enter the planner before any retrieval.

RAG runs only when one of these conditions is observed:

* a `knowledge_query` has confidence below
  `knowledge.retrieval_confidence_threshold` (default `0.60`);
* the user explicitly requests a source, citation, basis, audit, verification,
  or private-knowledge lookup;
* the planner returns `Decision(action="retrieve")`.

An explicit negative instruction such as “无需来源”, “不要核实”, “do not
cite sources”, or “no evidence needed” takes precedence over those keywords
and bypasses RAG for that turn.

The same normalized query is retrieved at most once per run. A repeated
planner request is stopped and recorded as a warning. A required lookup that
fails or returns no valid evidence produces `partial`, no citations, and an
explicit verification-unavailable answer. For a direct conversation,
`rag_hits=0` and an empty citation list are normal and do not mean that an
attempted retrieval failed.

### Model dispatch

The current configuration contains one local model (`llm.model`), not a pool
of specialist models. Dispatch therefore selects a graph stage and prompt,
not a different model endpoint:

* deterministic intent rules run first; the configured LLM is used only for
  ambiguous classification or when LLM intent classification is forced;
* general conversation and confident knowledge questions go directly to the
  response synthesizer;
* work and tool intents go to the protocol-constrained `LLMPlanner`, whose
  output must validate as a `Decision`;
* `retrieve` is the only planner action that reaches private RAG, and each run
  can search at most once;
* `call_tool` is allow-listed by the planner and then executed only through the
  MCP gateway; the model cannot invoke a Python tool directly;
* retrieved evidence returns through the planner and then the grounded
  synthesizer.

On each planner loop, the model receives the prior decision, bounded recent
tool results, the current observation, and the step count. This lets it decide
from an executed result instead of blindly repeating the same tool call.

In `--mock` mode there is no model planner. Evidence already retrieved can be
rendered without a model, but a work request that requires planning returns
`partial` with `planner_unavailable`; it is not reported as completed.

## Install and configuration

Install the pinned dependencies from `requirements.txt` (or install the
package in editable mode):

```text
python -m pip install -r requirements.txt
```

Copy `config/langgraph_agent.yaml` and adjust paths only when they remain
under the workspace.  The following environment variables are supported:

* `AEROSPACE_LANGGRAPH_CONFIG` selects a YAML file.
* `AEROSPACE_LOCAL_LLM_BASE_URL` overrides `llm.endpoint`.
* `AEROSPACE_LOCAL_LLM_MODEL` overrides `llm.model`.

The default Qwen endpoint is `http://127.0.0.1:8000/v1` and the default model
is `qwythos`.  The CLI checks `/models`; it never starts a model process.  If
the endpoint is unavailable, commands that need an LLM return exit code 2
and JSON error code `LLM_UNAVAILABLE`.  Use `--mock` for an offline,
deterministic run.

## CLI

All commands accept `--workspace`, `--config`, `--json`, `--mock`, and
`--thread`.  Action flags are mutually exclusive.  In JSON mode stdout is
exactly one JSON document; diagnostics are sent to stderr.

```text
python start_langgraph_agent.py --mock --task "two-body central gravity"
python start_langgraph_agent.py --mock --stream --task "two-body central gravity"
python start_langgraph_agent.py --mock --json --task "find launch constraints" --thread demo
python start_langgraph_agent.py --json --tools
python start_langgraph_agent.py --json --init-knowledge
python start_langgraph_agent.py --json --knowledge-status
python start_langgraph_agent.py --json --knowledge-graph reports/graph.html
python start_langgraph_agent.py --mock --json --checkpoint-history demo
python start_langgraph_agent.py --json --evolve demo
python start_langgraph_agent.py --json --evolve demo --approve-evolution
python start_langgraph_agent.py --json --evolve-due
python start_langgraph_agent.py --json --rollback EVOLUTION_ID
python start_langgraph_agent.py --json --export-schemas
```

`--knowledge-graph` writes sibling deterministic HTML and JSON files and
returns absolute paths.  `--export-schemas` writes the nine public protocol
schemas under `schemas/langgraph_agent/` using the versioned filenames
`agent-input-v1.json`, `agent-output-v1.json`, `decision-v1.json`,
`evidence-item-v1.json`, `tool-call-request-v1.json`,
`tool-call-response-v1.json`, `evolution-proposal-v1.json`,
`evolution-record-v1.json`, and `validation-result-v1.json`.

## MCP stdio

The configured transport is stdio.  A client may launch:

```text
python -m aerospace_agent.mcp.server
```

The server speaks MCP over stdin/stdout; keep logs on stderr.  The configured
command and arguments live in the `mcp` section of the YAML file.  In-process
tool calls are available to the deterministic mock runtime for tests.

Optional STK COM probing is disabled by default because COM startup can block
an unattended process.  Enable it only on a validated STK host with
`AEROSPACE_ENABLE_STK_COM_PROBE=1`; availability probes otherwise return a
bounded, explicit unavailable result.

## Checkpoints and threads

SQLite checkpoints default to `data/langgraph/checkpoints.sqlite`; tests and
short-lived clients can select the in-memory backend.  A thread id is the
stable conversation key.  `--checkpoint-history THREAD` returns graph-native
checkpoint ids, parent ids, next nodes, and metadata without exposing raw
runtime objects.  Checkpoint replay and resume are implemented by the agent
facade, not by the CLI.

## Evolution safeguards and rollback

Evolution proposals are validated before any write.  Target paths must be
relative, must stay below one of the configured roots (`knowledge`, `memory`,
`evolved_skills`, or `workflows/evolved` by default), and may not traverse a
symlink.  Each transaction journals state transitions, stores before/after
hashes, stages files, runs validators, and compensates on failure.  Unfinished
items are persisted to `memory/pending.md` rather than applied automatically.

Rollback is conditional: the current file hash must equal the recorded
post-commit hash.  A mismatch is a `conflict` and the file is left untouched.
An unknown id returns exit code 2 and error code `EVOLUTION_NOT_FOUND`.  A
successful rollback is recorded as `rolled_back` in the same journal.

The graph safety layer rejects non-finite orbital values and missing units,
reference frame, or time-scale metadata.  When declared, acceleration and
specific energy are checked against the two-body invariants.  High-risk MCP
tools and evolution writes require an approval callback (or an explicit
`human_approved=True` argument); without approval they are blocked rather
than executed.

When a local model synthesizes an answer from retrieved evidence, each
sentence is checked against a retrieved excerpt; an unsupported response is
discarded and the agent returns the evidence rendering with a grounding
warning. Direct turns that did not request private evidence are not presented
as RAG-grounded answers. This is a conservative provenance gate, not a
general-purpose semantic-entailment model.

Built distributions include the YAML configuration, six Wiki pages, and the
versioned protocol schemas under `share/aerospace-agent/`.

## Seed knowledge scope

`--init-knowledge` renders exactly six deterministic orbital-dynamics pages:
two-body dynamics, perturbed propagation, reference frames and time scales,
orbit determination and measurements, propagation validation, and
truth-to-sensor mapping.  The Wiki is the source of truth; RAG chunks and
knowledge-graph edges are derived only after all six source writes succeed.
Repeated initialization is idempotent and reports `created=0` when no source
page changed.

## Local WebUI

The first WebUI phase is local-only and uses the existing Agent Core boundary.
Start it with:

```text
python -m aerospace_agent.web --workspace .
```

The default address is `http://127.0.0.1:8765/`. The health endpoint is
`/api/v1/health` and the run channel is `/api/v1/ws`. For frontend development,
run `npm install` and `npm run dev` from `webui/`; Vite proxies `/api` to the
local backend. A production build runs `npm run build` and writes static assets
to `aerospace_agent/web/static/`.

This phase deliberately does not expose LAN access, token streaming,
cancellation, approval actions, file browsing, MCP tools, Apps, Skills, or
Automations. Human approval remains inside the existing Agent Core flow and is
shown as an interrupted run with a reason code.

## Test artifact hygiene

Automated tests write only below `.test-artifacts/`.  Successful acceptance
runs remove their own temporary fixture data; failed runs retain that single
directory for diagnosis.  Python bytecode and pytest caches are disabled in
the documented verification commands.  Runtime state under `data/langgraph/`
is a formal product artifact and is not mixed with test fixtures.
