# LangGraph Aerospace Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first LangGraph aerospace agent with versioned protocols, persistent conversation recovery, MCP tools, a six-seed Markdown Wiki/RAG/knowledge graph, and a reversible self-evolution backend.

**Architecture:** Keep LangGraph as a thin orchestration layer and inject four single-process domain services: context, knowledge, MCP, and evolution. Markdown is the auditable knowledge source; SQLite checkpoints and evolution transaction records persist execution and file-change state separately.

**Tech Stack:** Python 3.10–3.13, LangGraph 1.x, langchain-core 1.x, Pydantic 2, langgraph-checkpoint-sqlite, official MCP Python SDK, PyYAML, existing NumPy-based hybrid RAG, pytest.

**Required execution skills:** `@superpowers:test-driven-development` for every behavior change, `@superpowers:systematic-debugging` for failures, and `@superpowers:verification-before-completion` before claiming completion.

**Approved spec:** `docs/superpowers/specs/2026-07-10-langgraph-aerospace-agent-design.md`

**Worktree note:** Execute in the current workspace. The user-created `aerospace_agent/langgraph_agent/` tree and related RAG/MCP edits are untracked or modified only here; a new worktree would omit those facts. Stage only files named by the current task and never stage existing unrelated data/artifacts.

---

## File map

| Path | Responsibility |
| --- | --- |
| `requirements.txt`, `setup.py` | Declare the tested LangGraph/Pydantic/MCP runtime contract. |
| `config/langgraph_agent.yaml` | Default local model, runtime limits, paths, checkpointer, knowledge, MCP, and evolution settings. |
| `aerospace_agent/langgraph_agent/config.py` | Load YAML, apply explicit environment overrides, resolve and validate workspace paths. |
| `aerospace_agent/langgraph_agent/schema.py` | Versioned Pydantic boundary models and JSON Schema export. |
| `aerospace_agent/langgraph_agent/state.py` | Checkpoint-safe graph state and reducers only. |
| `aerospace_agent/langgraph_agent/services/context.py` | Essential/summary/recent context assembly and artifact offload. |
| `aerospace_agent/langgraph_agent/services/wiki.py` | Safe, deterministic Markdown page/index/log persistence. |
| `aerospace_agent/langgraph_agent/services/knowledge.py` | Facade for Wiki ingestion, hybrid RAG rebuild/search, and graph synchronization. |
| `aerospace_agent/langgraph_agent/services/graph_export.py` | Deterministic knowledge-graph JSON and self-contained HTML export. |
| `aerospace_agent/langgraph_agent/services/mcp_gateway.py` | MCP gateway protocol, stdio client, and explicit in-process fallback. |
| `aerospace_agent/langgraph_agent/services/evolution_policy.py` | Writable-root resolution, traversal/symlink protection, and eligibility policy. |
| `aerospace_agent/langgraph_agent/services/evolution_store.py` | Evolution directories, manifests, snapshots, hashes, permissions, and state journal. |
| `aerospace_agent/langgraph_agent/services/evolution_validators.py` | Markdown, Wiki index, Skill manifest, workflow YAML, and affected-test validators. |
| `aerospace_agent/langgraph_agent/services/evolution.py` | Transaction orchestration, conversation review, due scanning, commit, and rollback. |
| `aerospace_agent/langgraph_agent/cycle_detector.py` | Pure state fingerprint and thread-local loop decision helpers. |
| `aerospace_agent/langgraph_agent/nodes.py` | Thin state transformations that invoke injected services. |
| `aerospace_agent/langgraph_agent/graph.py` | StateGraph topology, routing, checkpointer compilation, optional interrupts. |
| `aerospace_agent/langgraph_agent/checkpointer.py` | SQLite/in-memory saver lifecycle and history/list helpers. |
| `aerospace_agent/langgraph_agent/agent.py` | Public run/stream/resume/history/evolve facade and error/status mapping. |
| `aerospace_agent/langgraph_agent/evolution.py` | Backward-compatible exports for the new evolution service. |
| `aerospace_agent/mcp/server.py` | Protocol-clean stdio MCP server; diagnostics go to stderr. |
| `start_langgraph_agent.py` | Terminal commands and JSON output. |
| `knowledge/` | Tracked six-page Markdown Wiki, index, and append-only log. |
| `schemas/langgraph_agent/` | Tracked exported JSON Schema documents. |
| `tests/langgraph_agent/conftest.py` | Workspace, settings, service, graph, and agent factories shared by focused tests. |
| `tests/langgraph_agent/` | Unit, graph integration, checkpoint, evolution, MCP, CLI, and Qwen tests. |
| `scripts/run_langgraph_acceptance.py` | Isolated terminal runner and deterministic JSON/Markdown evidence report generator. |
| `docs/LANGGRAPH_AGENT.md` | Operator guide and verified command examples. |

## Chunk 1: Protocols, configuration, and auditable knowledge

### Task 1: Runtime dependencies and validated configuration

**Files:**
- Modify: `requirements.txt`
- Modify: `setup.py`
- Modify: `.gitignore`
- Create: `config/langgraph_agent.yaml`
- Create: `aerospace_agent/langgraph_agent/config.py`
- Create: `tests/langgraph_agent/__init__.py`
- Create: `tests/langgraph_agent/test_config.py`

- [ ] **Step 1: Write failing dependency and configuration tests**

```python
import ast
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.config import AgentSettings, load_settings

REQUIRED = {
    "langgraph": ("1.0", "2.0"),
    "langgraph-checkpoint-sqlite": ("3.0", "4.0"),
    "langchain-core": ("1.0", "2.0"),
    "pydantic": ("2.0", "3.0"),
    "mcp": ("1.0", "2.0"),
}


def test_runtime_dependencies_are_bounded_in_requirements_and_setup():
    requirements = {
        line.split(">=")[0]: line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and ">=" in line
    }
    tree = ast.parse(Path("setup.py").read_text(encoding="utf-8"))
    setup_specs = {
        item.value.split(">=")[0]: item.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "install_requires"
        for item in node.value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    for name, (low, high) in REQUIRED.items():
        exact = f"{name}>={low},<{high}"
        assert requirements[name] == exact
        assert setup_specs[name] == exact


def test_settings_resolve_paths_inside_workspace(tmp_path):
    settings = load_settings(workspace=tmp_path)
    assert settings.knowledge.workspace == tmp_path / "knowledge"
    assert settings.checkpoint.path == tmp_path / "data/langgraph/checkpoints.sqlite"
    assert settings.knowledge.data_dir == tmp_path / "data/langgraph/rag"
    assert settings.context.artifacts_dir == tmp_path / "data/langgraph/artifacts"
    assert settings.evolution.data_dir == tmp_path / "data/langgraph/evolution"
    assert settings.knowledge.graph_output == tmp_path / "reports/knowledge_graph.html"
    assert settings.evolution.allowed_roots == [
        tmp_path / "knowledge", tmp_path / "memory",
        tmp_path / "evolved_skills", tmp_path / "workflows/evolved",
    ]


def test_settings_reject_path_escape(tmp_path):
    escaped = [
        {"knowledge": {"workspace": "../outside"}},
        {"knowledge": {"data_dir": "../outside"}},
        {"checkpoint": {"path": "../outside/checkpoints.sqlite"}},
        {"context": {"artifacts_dir": "../outside"}},
        {"evolution": {"data_dir": "../outside"}},
        {"evolution": {"allowed_roots": ["../outside"]}},
    ]
    for mapping in escaped:
        with pytest.raises(ValueError, match="workspace"):
            AgentSettings.from_mapping(mapping, workspace=tmp_path)


def test_explicit_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AEROSPACE_LANGGRAPH_CONFIG", str(tmp_path / "custom.yaml"))
    monkeypatch.setenv("AEROSPACE_LOCAL_LLM_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("AEROSPACE_LOCAL_LLM_MODEL", "test-model")
    (tmp_path / "custom.yaml").write_text("llm: {}\n", encoding="utf-8")
    settings = load_settings(workspace=tmp_path)
    assert settings.llm.endpoint == "http://127.0.0.1:9000/v1"
    assert settings.llm.model == "test-model"
```

- [ ] **Step 2: Run tests and confirm the missing module/config failure**

Run: `pytest tests/langgraph_agent/test_config.py -q`

Expected: FAIL because `langgraph_agent.config` and the dependency declarations do not exist.

- [ ] **Step 3: Declare bounded dependencies**

Add compatible bounds to both packaging surfaces:

```text
langgraph>=1.0,<2.0
langgraph-checkpoint-sqlite>=3.0,<4.0
langchain-core>=1.0,<2.0
pydantic>=2.0,<3.0
mcp>=1.0,<2.0
```

Preserve existing dependencies. Add `.superpowers/`, `/data/langgraph/`, `/memory/`, `/evolved_skills/`, and `/workflows/evolved/` to `.gitignore`; do not ignore `knowledge/`, `config/`, or `schemas/`.

- [ ] **Step 4: Add the complete default YAML**

Copy every field and value from spec section 12, plus these exact defaults: `context.artifacts_dir: data/langgraph/artifacts`, `evolution.data_dir: data/langgraph/evolution`, and `knowledge.graph_output: reports/knowledge_graph.html`. Do not add undeclared external-service settings.

- [ ] **Step 5: Implement nested settings models and YAML parsing**

Create `LLMSettings`, `RuntimeSettings`, `ContextSettings`, `KnowledgeSettings`, `CheckpointSettings`, `EvolutionSettings`, `MCPSettings`, and `AgentSettings`. `AgentSettings.from_mapping()` merges defaults with the provided mapping; `load_settings()` reads `AEROSPACE_LANGGRAPH_CONFIG` when set, otherwise `config/langgraph_agent.yaml`.

- [ ] **Step 6: Implement environment overrides and validate every writable path**

Apply only `AEROSPACE_LANGGRAPH_CONFIG`, `AEROSPACE_LOCAL_LLM_BASE_URL`, and `AEROSPACE_LOCAL_LLM_MODEL`. Resolve all paths listed in `test_settings_resolve_paths_inside_workspace`; use `Path.is_relative_to(workspace.resolve())` and raise `ValueError("path escapes workspace: <path>")` on escape.

- [ ] **Step 7: Install the project runtime**

Run: `python -m pip install -e ".[dev,mcp-server]"`

Expected: installation succeeds. If dependency download is blocked, request network escalation; do not replace the libraries with local stubs.

- [ ] **Step 8: Verify exact imports**

Run: `python -c "import langgraph, langchain_core, mcp, pydantic; from langgraph.checkpoint.sqlite import SqliteSaver; print('LANGGRAPH_RUNTIME_OK')"`

Expected: exit 0 and stdout contains `LANGGRAPH_RUNTIME_OK`.

- [ ] **Step 9: Run the focused tests**

Run: `pytest tests/langgraph_agent/test_config.py -q`

Expected: PASS.

- [ ] **Step 10: Commit only Task 1 files**

```powershell
git add requirements.txt setup.py .gitignore config/langgraph_agent.yaml aerospace_agent/langgraph_agent/config.py tests/langgraph_agent/__init__.py tests/langgraph_agent/test_config.py
git commit -m "build: declare langgraph agent runtime"
```

### Task 2: Versioned boundary protocols and checkpoint-safe state

**Files:**
- Modify: `aerospace_agent/langgraph_agent/schema.py`
- Modify: `aerospace_agent/langgraph_agent/state.py`
- Modify: `aerospace_agent/langgraph_agent/__init__.py`
- Create: `tests/langgraph_agent/test_schema_state.py`
- Create: `schemas/langgraph_agent/.gitkeep` initially; replace it with exported JSON in Task 9

- [ ] **Step 1: Write failing protocol tests**

```python
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aerospace_agent.langgraph_agent.schema import (
    ActionType, AgentInput, AgentOutput, Decision, EvidenceItem,
    EvolutionFileChange, EvolutionProposal, EvolutionRecord, RunStatus,
    ToolCallRequest, ToolCallResponse, export_json_schemas,
)
from aerospace_agent.langgraph_agent.state import create_initial_state


def test_agent_input_has_versioned_runtime_contract():
    value = AgentInput(user_message="Explain two-body dynamics", thread_id="t1")
    assert value.schema_version == "1.0.0"
    assert value.recursion_limit > value.max_steps
    assert value.run_id


def test_agent_output_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        AgentOutput(status="looks-good", answer="x")


def test_intermediate_protocol_constraints_and_independent_defaults():
    evidence = EvidenceItem(
        source_id="seed:two-body", page_path="knowledge/a.md",
        chunk_id="a:0", score=0.8, excerpt="central gravity",
    )
    decision = Decision(action=ActionType.RESPOND, rationale="evidence is enough")
    output_a = AgentOutput(status=RunStatus.SUCCESS, answer="x", citations=[evidence])
    output_b = AgentOutput(status=RunStatus.SUCCESS, answer="y")
    output_a.warnings.append("only-a")
    assert output_b.warnings == []
    assert decision.action == "respond"
    with pytest.raises(ValueError, match="score"):
        EvidenceItem(source_id="s", page_path="knowledge/a.md", chunk_id="c", score=2)
    with pytest.raises(ValueError, match="relative"):
        EvolutionFileChange(operation="update", path=Path("../outside"), content="x")


def test_tool_and_evolution_models_cover_success_and_failure():
    request = ToolCallRequest(tool_name="check_engine_availability")
    response = ToolCallResponse(tool_name=request.tool_name, status="success", result={})
    proposal = EvolutionProposal(thread_id="t", run_id="r", rationale="reuse", changes=[])
    record = EvolutionRecord(evolution_id="e", thread_id="t", run_id="r", status="proposed")
    assert response.status == "success"
    assert proposal.changes == []
    assert record.status == "proposed"


def test_state_round_trips_through_langgraph_jsonplus():
    state = create_initial_state("t1", "r1", max_steps=5)
    state["messages"] = [HumanMessage(content="hello")]
    assert "knowledge_service" not in state
    assert "db_connection" not in state
    serializer = JsonPlusSerializer()
    type_name, payload = serializer.dumps_typed(state)
    restored = serializer.loads_typed((type_name, payload))
    assert restored["thread_id"] == "t1"
    assert restored["messages"][0].content == "hello"


def test_json_schema_export_contains_every_public_protocol():
    schemas = export_json_schemas()
    required = {
        "AgentInput", "AgentOutput", "Decision", "EvidenceItem",
        "ToolCallRequest", "ToolCallResponse", "EvolutionProposal", "EvolutionRecord",
        "OrbitState", "KeplerianOrbitState", "OrbitDesignRequest", "OrbitDesignResponse",
        "RagQueryRequest", "RagQueryResponse",
    }
    assert required <= schemas.keys()
    assert all(schema.get("type") == "object" for schema in schemas.values())
```

- [ ] **Step 2: Run tests and verify protocol failures**

Run: `pytest tests/langgraph_agent/test_schema_state.py -q`

Expected: FAIL on missing `Decision`, `EvidenceItem`, `EvolutionProposal`, `EvolutionRecord`, and `RunStatus`.

- [ ] **Step 3: Implement status/action enums and top-level Agent protocols**

Define `RunStatus`, `ActionType`, and `IntentType` string enums. `AgentInput` has `schema_version`, non-empty `user_message/thread_id/run_id`, `mode`, `max_steps`, `recursion_limit`, and `context`; validate `recursion_limit > max_steps`. `AgentOutput` has the exact spec fields and uses `Field(default_factory=list/dict)` for every collection.

- [ ] **Step 4: Implement evidence, tool, decision, and evolution protocols**

`EvidenceItem` validates score `[0,1]`, relative Wiki `page_path`, non-empty source/chunk IDs, bounded excerpt, and source metadata. `Decision` requires `action`, `rationale`, `next_action`, and `tool_args`; it requires a `ToolCallRequest` only for `call_tool`. `ToolCallResponse` requires `error` for non-success. `EvolutionFileChange` accepts only `create/update/delete`, rejects absolute/`..` paths, and requires content except for delete. `EvolutionProposal` requires thread/run/checkpoint source, unfinished-item list, and required validation names. `EvolutionRecord` uses exactly `proposed | staged | backed_up | validating | committed | rollback_requested | validation_failed | commit_failed | rolled_back | conflict` and stores proposal, before/after manifest entries, validation results, and report path.

- [ ] **Step 5: Implement `export_json_schemas()` from the public model list**

Return a deterministic mapping for every model asserted in the test, including all existing public orbit/RAG models listed above; do not silently drop backward-compatible exports. Do not write files in this function; Task 9 owns artifact generation.

- [ ] **Step 6: Refactor state to contain only serialized values**

Keep `messages: Annotated[Sequence[BaseMessage], add_messages]`. Add `run_id`, `decision`, `evidence`, `tool_requests`, `tool_results`, `artifact_refs`, `state_fingerprints`, `step_count`, `intervention_count`, `termination_reason`, and `checkpoint_id`. Remove service instances from state and the redundant state `max_recursion_depth`; retain `max_steps` as a business budget.

- [ ] **Step 7: Run focused tests**

Run: `pytest tests/langgraph_agent/test_schema_state.py -q`

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```powershell
git add aerospace_agent/langgraph_agent/schema.py aerospace_agent/langgraph_agent/state.py aerospace_agent/langgraph_agent/__init__.py tests/langgraph_agent/test_schema_state.py schemas/langgraph_agent/.gitkeep
git commit -m "feat: define langgraph agent protocols"
```

### Task 3: Six-seed Markdown Wiki, persistent RAG, and graph export

**Files:**
- Create: `aerospace_agent/langgraph_agent/services/__init__.py`
- Create: `aerospace_agent/langgraph_agent/services/wiki.py`
- Create: `aerospace_agent/langgraph_agent/services/knowledge.py`
- Create: `aerospace_agent/langgraph_agent/services/graph_export.py`
- Modify: `aerospace_agent/rag/orbit_dynamics.py`
- Create: `tests/langgraph_agent/conftest.py`
- Create: `tests/langgraph_agent/test_knowledge_service.py`
- Create: `knowledge/index.md`
- Create: `knowledge/log.md`
- Create: `knowledge/orbital-dynamics/two-body-orbital-dynamics.md`
- Create: `knowledge/orbital-dynamics/reference-frames-and-time-scales.md`
- Create: `knowledge/orbital-dynamics/perturbed-orbit-propagation.md`
- Create: `knowledge/orbital-dynamics/orbit-determination-and-measurements.md`
- Create: `knowledge/orbital-dynamics/propagation-validation.md`
- Create: `knowledge/orbital-dynamics/truth-to-sensor-mapping.md`

- [ ] **Step 1: Write failing Wiki idempotency and evidence tests**

```python
import json
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService
from aerospace_agent.langgraph_agent.services.wiki import WikiStore


def test_seed_wiki_is_idempotent_and_searchable(tmp_path):
    service = KnowledgeService(workspace=tmp_path)
    first = service.initialize_seed_wiki()
    index_before = (tmp_path / "knowledge/index.md").read_bytes()
    log_before = (tmp_path / "knowledge/log.md").read_bytes()
    second = service.initialize_seed_wiki()
    assert first.created == 6
    assert second.created == 0
    assert len(list((tmp_path / "knowledge/orbital-dynamics").glob("*.md"))) == 6
    assert (tmp_path / "knowledge/index.md").read_bytes() == index_before
    assert (tmp_path / "knowledge/log.md").read_bytes() == log_before
    assert log_before.count(b"| ingest |") == 6

    evidence = service.search("two-body central gravity", top_k=3)
    assert evidence
    assert evidence[0].page_path.endswith("two-body-orbital-dynamics.md")
    assert evidence[0].chunk_id
    assert 0 <= evidence[0].score <= 1
    assert evidence[0].metadata["content_sha256"]
    assert evidence[0].metadata["page_id"].startswith("seed:")
    assert evidence[0].metadata["page_path"].endswith(".md")

    reloaded = KnowledgeService(workspace=tmp_path)
    assert reloaded.search("two-body central gravity", top_k=1)[0].page_path == evidence[0].page_path


def test_every_seed_page_has_valid_cross_references(tmp_path):
    service = KnowledgeService(workspace=tmp_path)
    service.initialize_seed_wiki()
    pages = list((tmp_path / "knowledge/orbital-dynamics").glob("*.md"))
    assert len(pages) == 6
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert "## Related pages" in text
        for relative in service.wiki.extract_links(page):
            assert (page.parent / relative).resolve().is_file()


def test_wiki_store_rejects_traversal(tmp_path):
    store = WikiStore(tmp_path / "knowledge")
    with pytest.raises(ValueError, match="knowledge root"):
        store.write_relative(Path("../escape.md"), "# escape")


def test_graph_export_links_back_to_wiki(tmp_path):
    service = KnowledgeService(workspace=tmp_path)
    service.initialize_seed_wiki()
    output = service.export_graph(tmp_path / "reports/graph.html")
    html = output.html_path.read_text(encoding="utf-8")
    graph = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert "two-body-orbital-dynamics.md" in html
    assert any(edge["relation"] == "related_to" for edge in graph["edges"])
    wiki_nodes = [n for n in graph["nodes"] if n["type"] == "wiki_page"]
    assert len(wiki_nodes) == 6
    assert all(n["id"].startswith("wiki:") and n["page_path"].endswith(".md") for n in wiki_nodes)
    assert all(n["content_sha256"] for n in wiki_nodes)
    assert all(n["page_path"] in html for n in wiki_nodes)
```

In `tests/langgraph_agent/conftest.py`, add executable shared fixtures:

```python
from pathlib import Path
import pytest
from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def knowledge_service(workspace: Path) -> KnowledgeService:
    service = KnowledgeService(workspace=workspace)
    service.initialize_seed_wiki()
    return service
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/langgraph_agent/test_knowledge_service.py -q`

Expected: FAIL because `KnowledgeService` does not exist.

- [ ] **Step 3: Add stable IDs, slugs, and related-page metadata to the six seeds**

Do not change the factual seed text. Add deterministic slugs and explicit relationships. The first seed entry is exactly:

```python
{
    "topic": "two_body_dynamics",
    "slug": "two-body-orbital-dynamics",
    "title": "Two-body orbital dynamics",
    "text": (
        "Two-body dynamics models spacecraft motion under a central gravity field. "
        "The governing acceleration is -mu r / |r|^3. Keplerian elements are useful "
        "for compact orbit description, while Cartesian states are preferred for "
        "numerical propagation and covariance operations. Assumptions: point-mass "
        "gravity, no drag, no third-body perturbation, no finite burn."
    ),
    "related_topics": ["perturbations", "validation"],
}
```

Add the corresponding slug and at least one valid `related_topics` entry to each of the other five existing seeds.

- [ ] **Step 4: Implement traversal-safe Wiki page persistence in `wiki.py`**

`WikiStore.resolve_relative()` rejects absolute paths, `..`, and resolved paths outside its root. `write_relative()` compares SHA256 and returns `created | updated | unchanged`; changed writes use a temporary sibling followed by `os.replace()`.

- [ ] **Step 5: Implement deterministic page rendering, index, and append-only log**

Each page contains title, `Source: built-in seed corpus`, stable page ID, content SHA256, seed body, and `## Related pages` relative links. Sort index entries by category/slug. Append one dated ingest/update line only for changed pages; never rewrite prior log bytes.

- [ ] **Step 6: Implement `KnowledgeService.initialize_seed_wiki()`**

Render all six pages, then rebuild the index from parsed Wiki pages. Return a typed summary with created/updated/unchanged counts and paths. If a write fails, do not rebuild derived RAG.

- [ ] **Step 7: Implement deterministic full RAG rebuild and reload**

Instantiate `AerospaceRAG(data_dir=<workspace>/data/langgraph/rag, autoload=False, auto_default_knowledge=False)`. Because the current `AerospaceRAG.index()` drops metadata kwargs, call its `kb.index_text(chunk, source=..., metadata={page_id,page_path,chunk_id,content_sha256})` directly, then reindex/save. Add page nodes and `related_to` edges. `search()` returns `EvidenceItem` values from `query_results()`; do not return formatted strings to graph nodes.

- [ ] **Step 8: Synchronize all six Wiki nodes and cross-reference edges**

Use stable node IDs `wiki:<page_id>`. Attach `type=wiki_page`, `page_path`, `content_sha256`, and aliases. Parse relative links and add `related_to` only when both endpoint pages exist. Save and reload the graph as part of the persistence test; assert the six node IDs, page paths, hashes, and edge relations survive reload.

- [ ] **Step 9: Implement deterministic JSON and self-contained HTML export in `graph_export.py`**

Write sibling `graph.json` and requested HTML. Sort nodes/edges, embed the exact JSON payload, and render Wiki nodes as clickable relative `page_path` links. Reuse the existing knowledge-cloud styling only if it preserves deterministic payload and links.

- [ ] **Step 10: Generate the tracked six-page Wiki using the service**

Run: `python -c "from pathlib import Path; from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService; print(KnowledgeService(Path.cwd()).initialize_seed_wiki().model_dump_json())"`

Expected: JSON reports `created=6`; six exact files, `index.md`, and `log.md` exist. Review generated Markdown; never hand-edit generated hashes.

- [ ] **Step 11: Run focused and existing RAG tests**

Run: `pytest tests/langgraph_agent/test_knowledge_service.py tests/test_rag_multisource.py -q`

Expected: PASS.

- [ ] **Step 12: Commit Task 3**

```powershell
git add aerospace_agent/langgraph_agent/services aerospace_agent/rag/orbit_dynamics.py tests/langgraph_agent/conftest.py tests/langgraph_agent/test_knowledge_service.py knowledge
git commit -m "feat: add orbital dynamics markdown wiki"
```

## Chunk 2: Context, MCP, graph runtime, and checkpoints

### Task 4: Context assembly, artifacts, and thread-local loop detection

**Files:**
- Create: `aerospace_agent/langgraph_agent/services/context.py`
- Rewrite: `aerospace_agent/langgraph_agent/cycle_detector.py`
- Create: `tests/langgraph_agent/test_context_cycle.py`

- [ ] **Step 1: Write failing context and cycle-isolation tests**

```python
from langchain_core.messages import HumanMessage, SystemMessage

from aerospace_agent.langgraph_agent.cycle_detector import evaluate_cycle
from aerospace_agent.langgraph_agent.services.context import ContextService


def test_context_preserves_current_instruction_and_offloads_large_result(tmp_path):
    service = ContextService(tmp_path, max_tokens=100, recent_turns=2, artifact_chars=50)
    result = service.assemble(
        messages=[
            SystemMessage(content="SYSTEM CONSTRAINT: use SI units"),
            HumanMessage(content="old " * 100),
            HumanMessage(content="CURRENT"),
        ],
        tool_results=[{"payload": "x" * 500}],
    )
    assert "CURRENT" in result.prompt
    assert "SYSTEM CONSTRAINT" in result.prompt
    assert result.estimated_tokens <= 100
    assert len(result.recent_messages) <= 4  # two user/assistant turns
    assert len(result.artifact_refs) == 1
    ref = result.artifact_refs[0]
    assert ref.path.is_relative_to(tmp_path)
    assert len(ref.sha256) == 64
    assert ref.media_type == "application/json"
    assert ref.summary
    assert ref.path.read_text(encoding="utf-8")
    assert "x" * 500 not in result.prompt


def test_cycle_history_is_owned_by_state_not_detector_instance():
    a = {"state_fingerprints": [], "step_count": 0, "intervention_count": 0}
    b = {"state_fingerprints": [], "step_count": 0, "intervention_count": 0}
    for _ in range(3):
        a.update(evaluate_cycle(a, action="retrieve", payload={"q": "same"}, max_repeats=3))
    assert a["intervention_count"] == 1
    assert b["intervention_count"] == 0


def test_cycle_fingerprint_distinguishes_tool_name_and_target():
    empty = {"state_fingerprints": [], "step_count": 0, "intervention_count": 0}
    first = evaluate_cycle(empty, action="call_tool", tool_name="propagate_orbit", target="leo", payload={"x": 1}, max_repeats=3)
    second = evaluate_cycle({**empty, **first}, action="call_tool", tool_name="propagate_orbit", target="moon", payload={"x": 1}, max_repeats=3)
    assert second["state_fingerprints"][-1] != first["state_fingerprints"][-1]
    other_tool = evaluate_cycle({**empty, **first}, action="call_tool", tool_name="convert_time", target="leo", payload={"x": 1}, max_repeats=3)
    assert other_tool["state_fingerprints"][-1] != first["state_fingerprints"][-1]
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/langgraph_agent/test_context_cycle.py -q`

Expected: FAIL on missing service and pure cycle API.

- [ ] **Step 3: Implement bounded essential/summary/recent assembly**

Adapt existing token estimation rather than duplicating it. Always keep system constraints and the current user instruction, plus at most `recent_turns * 2` recent messages. Summarize older content deterministically when no LLM summarizer is configured; iteratively trim the oldest summary item until `estimated_tokens <= max_tokens`.

- [ ] **Step 4: Implement content-addressed artifact offload**

Canonicalize large JSON with sorted keys, hash UTF-8 bytes, write `data/langgraph/artifacts/<sha256>.json`, and return an `ArtifactRef(path, sha256, media_type="application/json", summary)`. Verify an existing file's hash before reuse.

- [ ] **Step 5: Replace mutable `CycleDetector` counters with pure state updates**

Keep a compatibility `CycleDetector(max_repeats: int, max_steps: int)` facade, but make `check(state: AerospaceAgentState) -> tuple[bool, str, dict[str, object]]` derive all counts from `state`. Fingerprint normalized action, tool name, target, sorted parameter summary, observation digest/summary, and intent; do not include monotonically increasing depth in the fingerprint. `evaluate_cycle(state, action, tool_name, target, payload, max_repeats)` returns only a state delta and never stores counters on the detector instance.

- [ ] **Step 6: Run tests**

Run: `pytest tests/langgraph_agent/test_context_cycle.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add aerospace_agent/langgraph_agent/services/context.py aerospace_agent/langgraph_agent/cycle_detector.py tests/langgraph_agent/test_context_cycle.py
git commit -m "feat: add bounded context and loop guards"
```

### Task 5: Protocol-clean MCP server and gateway

**Files:**
- Modify: `aerospace_agent/mcp/server.py`
- Create: `aerospace_agent/langgraph_agent/services/mcp_gateway.py`
- Modify: `tests/langgraph_agent/conftest.py`
- Create: `tests/langgraph_agent/test_mcp_gateway.py`
- Modify only if required by failing consistency test: `aerospace_agent/mcp/tools/__init__.py`

- [ ] **Step 1: Write failing server purity and gateway tests**

```python
import pytest

from aerospace_agent.langgraph_agent.schema import ToolCallRequest
from aerospace_agent.langgraph_agent.services.mcp_gateway import InProcessMCPGateway
from aerospace_agent.mcp.server import _wrap_all_tools


def test_inprocess_gateway_lists_and_calls_schema_valid_tools():
    gateway = InProcessMCPGateway(_wrap_all_tools())
    names = {tool.name for tool in gateway.list_tools()}
    assert "check_engine_availability" in names
    response = gateway.call_tool(ToolCallRequest(tool_name="check_engine_availability"))
    assert response.status == "success"


def test_missing_required_arguments_are_rejected_before_call():
    gateway = InProcessMCPGateway(_wrap_all_tools())
    response = gateway.call_tool(ToolCallRequest(tool_name="propagate_orbit", arguments={}))
    assert response.status == "invalid_arguments"
    assert "initial_state_dict" in response.error
```

Add executable stdio/lifecycle tests:

```python
import asyncio
import sys

from aerospace_agent.langgraph_agent.services.mcp_gateway import (
    StdioMCPGateway, create_mcp_gateway,
)


def new_stdio_gateway():
    return StdioMCPGateway(
        command=sys.executable, args=["-m", "aerospace_agent.mcp.server"], timeout=30,
    )


def test_stdio_gateway_initialize_list_call_and_close():
    gateway = new_stdio_gateway()
    try:
        assert "check_engine_availability" in {t.name for t in gateway.list_tools()}
        result = gateway.call_tool(ToolCallRequest(tool_name="check_engine_availability"))
        assert result.status == "success"
    finally:
        gateway.close()
    assert gateway.closed is True


def test_sync_gateway_call_is_safe_while_caller_event_loop_is_running():
    async def caller():
        gateway = new_stdio_gateway()
        try:
            return gateway.list_tools()
        finally:
            gateway.close()
    assert asyncio.run(caller())


def test_explicit_inprocess_fallback_emits_warning(mcp_settings):
    gateway, warnings = create_mcp_gateway(
        mcp_settings, allow_inprocess_fallback=True, force_stdio_failure=True,
    )
    assert isinstance(gateway, InProcessMCPGateway)
    assert warnings == ["MCP stdio unavailable; using explicit in-process fallback"]
```

Add this fixture to `tests/langgraph_agent/conftest.py` before collecting the tests:

```python
@pytest.fixture
def mcp_settings(workspace):
    return load_settings(workspace=workspace).mcp
```

- [ ] **Step 2: Run tests and observe current stdio failure**

Run: `pytest tests/langgraph_agent/test_mcp_gateway.py -q`

Expected: FAIL because no gateway exists; the stdio test may additionally expose startup output and `stdio_server` scope defects.

- [ ] **Step 3: Fix the MCP server boundary**

Import `stdio_server` inside `_run_mcp_server`. In stdio mode, route startup diagnostics to `sys.stderr`. Return official `TextContent` values from `call_tool`, with JSON text containing structured status. Keep CLI mode human-readable.

- [ ] **Step 4: Implement and test the in-process gateway**

Define synchronous `list_tools()` and `call_tool()` protocols. `InProcessMCPGateway` validates required fields from tool definitions and returns `ToolCallResponse(status="invalid_arguments")` without calling the handler.

- [ ] **Step 5: Implement stdio gateway lifecycle on a dedicated loop thread**

Start one private event loop thread per gateway, create/close the official MCP session on that loop, and submit sync calls with `run_coroutine_threadsafe`. `close()` is idempotent, closes session/subprocess, stops the loop, and joins the thread. This avoids nested `asyncio.run()` even when the caller already has an event loop.

- [ ] **Step 6: Implement explicit fallback construction**

Default construction raises a structured MCP-unavailable error and an Agent run maps it to `tool_unavailable` without invoking a handler. Only `allow_inprocess_fallback=True` may return the fallback, together with the exact warning asserted above. Add a test asserting default construction does not call any in-process handler.

- [ ] **Step 7: Run MCP tests and existing schema tests**

Run: `pytest tests/langgraph_agent/test_mcp_gateway.py tests/test_mcp_schema_framework.py aerospace_agent/mcp/tests -q`

Expected: PASS, with optional engine tests returning structured unavailable results rather than crashing.

- [ ] **Step 8: Commit Task 5 without unrelated MCP edits**

```powershell
git add aerospace_agent/mcp/server.py aerospace_agent/langgraph_agent/services/mcp_gateway.py tests/langgraph_agent/test_mcp_gateway.py
git add aerospace_agent/mcp/tools/__init__.py  # only if this task changed it
git commit -m "fix: expose protocol-clean aerospace mcp"
```

### Task 6: Thin LangGraph nodes and terminating topology

**Files:**
- Rewrite: `aerospace_agent/langgraph_agent/nodes.py`
- Rewrite: `aerospace_agent/langgraph_agent/graph.py`
- Modify: `aerospace_agent/langgraph_agent/router.py`
- Modify: `tests/langgraph_agent/conftest.py`
- Create: `tests/langgraph_agent/test_graph_runtime.py`

- [ ] **Step 1: Write failing deterministic graph tests**

```python
from langgraph.checkpoint.memory import InMemorySaver

from aerospace_agent.langgraph_agent.graph import build_aerospace_graph


def test_knowledge_query_returns_cited_answer_without_tool_loop(services):
    graph = build_aerospace_graph(services=services, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "k1"}, "recursion_limit": 20}
    state = graph.invoke(services.initial_input("What is two-body dynamics?"), config)
    assert state["status"] == "success"
    assert state["evidence"]
    assert state["step_count"] < 10
    assert state["metrics"]["rag_hits"] >= 1
    assert state["metrics"]["node_timings_ms"]
    assert state["run_id"] and state["thread_id"] == "k1"


def test_repeated_decision_terminates_as_cycle(services_with_repeating_planner):
    graph = build_aerospace_graph(
        services=services_with_repeating_planner, checkpointer=InMemorySaver()
    )
    state = graph.invoke(
        services_with_repeating_planner.initial_input("repeat"),
        {"configurable": {"thread_id": "loop"}, "recursion_limit": 30},
    )
    assert state["status"] == "cycle_detected"
    assert state["intervention_count"] == 1
    assert state["termination_reason"] == "repeated_state_after_intervention"


def test_business_step_budget_returns_partial_with_checkpointable_state(services_with_repeating_planner):
    graph = build_aerospace_graph(
        services=services_with_repeating_planner,
        checkpointer=InMemorySaver(),
        cycle_max_repeats=99,
    )
    state = graph.invoke(
        services_with_repeating_planner.initial_input("repeat", max_steps=2),
        {"configurable": {"thread_id": "budget"}, "recursion_limit": 30},
    )
    assert state["status"] == "partial"
    assert state["step_count"] == 2
    assert state["termination_reason"] == "max_steps"


def test_tool_failure_is_categorized_and_observable(services_with_failing_tool):
    graph = build_aerospace_graph(
        services=services_with_failing_tool, checkpointer=InMemorySaver()
    )
    state = graph.invoke(
        services_with_failing_tool.initial_input("call failing tool"),
        {"configurable": {"thread_id": "tool-error"}, "recursion_limit": 20},
    )
    assert state["errors"][0]["category"] == "tool_error"
    assert state["tool_results"][0]["status"] == "error"
    assert state["metrics"]["tool_duration_ms"] >= 0
```

Extend `tests/langgraph_agent/conftest.py` with complete deterministic doubles and bundles before running the tests:

```python
from dataclasses import dataclass, replace
from aerospace_agent.langgraph_agent.schema import ActionType, Decision, ToolCallRequest
from aerospace_agent.langgraph_agent.services.context import ContextService
from aerospace_agent.langgraph_agent.services.mcp_gateway import InProcessMCPGateway


class RulePlanner:
    def __init__(self, decision=None):
        self.decision = decision or Decision(action=ActionType.RESPOND, rationale="use evidence")
    def plan(self, state):
        return self.decision


@pytest.fixture
def services(workspace, knowledge_service):
    return make_services(
        workspace=workspace,
        knowledge=knowledge_service,
        context=ContextService(workspace),
        planner=RulePlanner(),
        model_name="deterministic-test",
    )


@pytest.fixture
def services_with_repeating_planner(services):
    decision = Decision(
        action=ActionType.CALL_TOOL, rationale="repeat",
        tool_request=ToolCallRequest(tool_name="check_engine_availability"),
    )
    return replace(services, planner=RulePlanner(decision))


@pytest.fixture
def services_with_failing_tool(services):
    decision = Decision(
        action=ActionType.CALL_TOOL, rationale="exercise error",
        tool_request=ToolCallRequest(tool_name="failing_tool"),
    )
    return replace(
        services,
        planner=RulePlanner(decision),
        mcp_gateway=InProcessMCPGateway({"failing_tool": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}),
    )
```

`make_services()` is a test helper returning the exact service bundle type exported by `graph.py`; its `initial_input()` calls `create_initial_state()` and adds one `HumanMessage`.

Define it before collecting tests: add a frozen `ServiceBundle` dataclass to `tests/langgraph_agent/conftest.py` with fields `knowledge`, `context`, `planner`, `mcp_gateway`, `llm`, `model_name`, and `endpoint`; implement `make_services(...) -> ServiceBundle` and `ServiceBundle.initial_input(message, max_steps=15)` as the deterministic graph input factory. The production graph exports the same field names, so tests cannot hide missing dependencies behind an untyped dictionary.

- [ ] **Step 2: Run tests and verify old graph behavior fails**

Run: `pytest tests/langgraph_agent/test_graph_runtime.py -q`

Expected: FAIL because current nodes use empty tool arguments, formatted RAG strings, and a shared detector.

- [ ] **Step 3: Implement input/context/classify/retrieve thin nodes**

Each node accepts state and injected services, calls one service, and returns a state delta. Implement input validation, context hydration, classify/plan, and retrieve first. Convert Pydantic values with `model_dump(mode="json")` before writing state and record node timing by name.

- [ ] **Step 4: Implement decide/execute/validate/evaluate thin nodes**

`execute` calls only the request from `Decision`; it never invents empty arguments. `validate_observation` categorizes protocol/tool/retrieval errors. `evaluate` applies one cycle intervention, then cycle termination, and returns `partial` at `max_steps`.

- [ ] **Step 5: Implement synthesis and persisted run metrics**

Synthesis receives evidence excerpts and emits citations unchanged. `persist_outcome` finalizes run/thread IDs, endpoint/model name, node timings, tool duration, RAG hit count, cycle interventions, step count, total duration, final status, warnings, and categorized errors.

- [ ] **Step 6: Build explicit conditional edges**

Knowledge evidence can route directly to synthesis. Tool calls loop through observation/evaluation. `continue` is allowed only with remaining business steps and a changed fingerprint. Add optional `interrupt_before`/`interrupt_after` compile arguments for checkpoint/human-in-loop tests.

- [ ] **Step 7: Leave runtime recursion errors to the facade boundary**

Nodes handle business limits; `GraphRecursionError` remains visible to `LangGraphAerospaceAgent`, which maps it using the last checkpoint in Task 7.

- [ ] **Step 8: Run focused tests**

Run: `pytest tests/langgraph_agent/test_graph_runtime.py -q`

Expected: PASS.

- [ ] **Step 9: Commit Task 6**

```powershell
git add aerospace_agent/langgraph_agent/nodes.py aerospace_agent/langgraph_agent/graph.py aerospace_agent/langgraph_agent/router.py tests/langgraph_agent/conftest.py tests/langgraph_agent/test_graph_runtime.py
git commit -m "feat: build terminating aerospace state graph"
```

### Task 7: SQLite conversations, resume, history, and facade lifecycle

**Files:**
- Rewrite: `aerospace_agent/langgraph_agent/checkpointer.py`
- Rewrite: `aerospace_agent/langgraph_agent/agent.py`
- Modify: `aerospace_agent/langgraph_agent/__init__.py`
- Modify: `tests/langgraph_agent/conftest.py`
- Create: `tests/langgraph_agent/test_checkpoint_resume.py`

- [ ] **Step 1: Write failing multi-instance and interrupted-resume tests**

```python
def test_same_thread_survives_agent_recreation(agent_factory):
    first = agent_factory(checkpoint_backend="sqlite")
    first.run("What is two-body dynamics?", thread_id="persist")
    first.close()

    second = agent_factory(checkpoint_backend="sqlite")
    snapshot = second.get_conversation_state("persist")
    assert len(snapshot.values["messages"]) >= 2
    follow_up = second.run("What assumptions were stated?", thread_id="persist")
    assert follow_up.checkpoint_id


def test_resume_runs_from_saved_next_node(agent_factory):
    agent = agent_factory(interrupt_before=["synthesize"])
    result = agent.run("Explain validation", thread_id="resume")
    assert result.status == "interrupted"
    before = agent.get_conversation_state("resume")
    assert before.next == ("synthesize",)
    resumed = agent.resume_execution("resume")
    assert resumed.status == "success"


def test_thread_listing_history_order_replay_and_fork_isolation(agent_factory):
    agent = agent_factory()
    agent.run("What is two-body dynamics?", thread_id="source")
    agent.run("State one assumption", thread_id="source")
    history = agent.get_checkpoint_history("source")
    assert history
    assert [item.created_at for item in history] == sorted(
        [item.created_at for item in history], reverse=True
    )
    assert "source" in agent.list_conversations()
    source_before = agent.get_conversation_state("source").values
    fork = agent.fork_from_checkpoint(
        source_thread_id="source",
        checkpoint_id=history[-2].config["configurable"]["checkpoint_id"],
        new_thread_id="forked",
    )
    assert fork.thread_id == "forked"
    assert agent.get_conversation_state("source").values == source_before
    assert "forked" in agent.list_conversations()
    config = {"configurable": {"thread_id": "source"}}
    assert agent.get_state(config).values == source_before
    assert list(agent.get_state_history(config))


def test_graph_recursion_error_returns_last_checkpoint(agent_factory, services_with_repeating_planner):
    agent = agent_factory(
        service_bundle=services_with_repeating_planner,
        max_steps=50,
        recursion_limit=2,
        cycle_max_repeats=99,
    )
    output = agent.run("repeat", thread_id="recursion")
    assert output.status == "limit_reached"
    assert output.checkpoint_id
    assert output.errors[0].category == "graph_recursion_limit"
    assert output.metrics["model_name"] == "deterministic-test"


def test_checkpoint_write_failure_is_not_reported_as_saved(agent_factory, failing_checkpointer):
    agent = agent_factory(checkpointer=failing_checkpointer)
    output = agent.run("persist failure", thread_id="write-failure")
    assert output.status == "error"
    assert output.errors[0].category == "checkpoint_write_error"
    assert output.checkpoint_id is None
```

Extend `tests/langgraph_agent/conftest.py` with the concrete factory:

```python
from langgraph.checkpoint.memory import InMemorySaver
from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
from aerospace_agent.langgraph_agent.config import load_settings


@pytest.fixture
def agent_factory(workspace, services):
    created = []
    def factory(service_bundle=None, **overrides):
        base = load_settings(workspace=workspace)
        runtime = base.runtime.model_copy(update={
            key: value for key, value in overrides.items()
            if key in {"max_steps", "recursion_limit", "cycle_max_repeats"}
        })
        settings = base.model_copy(update={"runtime": runtime})
        agent = LangGraphAerospaceAgent(
            settings=settings,
            services=service_bundle or services,
            interrupt_before=overrides.get("interrupt_before"),
            checkpoint_backend=overrides.get("checkpoint_backend", "sqlite"),
            checkpoint_db_path=overrides.get(
                "checkpoint_db_path", workspace / "data/langgraph/checkpoints.sqlite"
            ),
            checkpointer=overrides.get("checkpointer"),
        )
        created.append(agent)
        return agent
    yield factory
    for agent in created:
        agent.close()


class FailingCheckpointer(InMemorySaver):
    def put(self, *args, **kwargs):
        raise OSError("checkpoint write")
    def put_writes(self, *args, **kwargs):
        raise OSError("checkpoint write")


@pytest.fixture
def failing_checkpointer():
    return FailingCheckpointer()
```

- [ ] **Step 2: Run tests and verify current fake resume fails**

Run: `pytest tests/langgraph_agent/test_checkpoint_resume.py -q`

Expected: FAIL because current `resume()` only checks raw SQLite rows and each run resets full state.

- [ ] **Step 3: Implement saver lifecycle and latest-state access**

Use the configured new database path. Hold the `SqliteSaver.from_conn_string()` context for the Agent lifetime and close it from `close()`, `__enter__/__exit__`, and best-effort destructor. Prefer `graph.get_state()`/`get_state_history()` over raw schema assumptions.

- [ ] **Step 4: Distinguish a new turn from resume execution**

For an existing completed thread, invoke only the new message and run metadata so reducers retain history. Every `invoke`/`stream` call must put `thread_id` under `configurable` and `recursion_limit` at the top level. For `snapshot.next`, `resume_execution()` invokes the saved graph without constructing a new initial state. Return `interrupted` when graph ends with pending `next` nodes.

- [ ] **Step 5: Add graph-native state aliases, thread listing, and ordered checkpoint history**

Expose both `get_state(config)`/`get_state_history(config)` aliases required by the public contract and the descriptive `get_conversation_state()`/`get_checkpoint_history()` helpers. List distinct thread IDs through saver records without hard-coding obsolete columns. Return graph-native `StateSnapshot` history, newest first, and include checkpoint IDs in public summaries.

- [ ] **Step 6: Add replay/fork without mutating source history**

Expose latest state, ordered history, and replay from a selected checkpoint config. A fork must use a new thread ID and copy only serialized state; never mutate the source history.

- [ ] **Step 7: Map `GraphRecursionError` to evidence-backed output**

Read the latest snapshot, return `limit_reached`, retain citations/tool results already checkpointed, and add the exception to errors. Do not return a generic success.

- [ ] **Step 8: Verify facade observability fields**

Add assertions to the same tests for run/thread/checkpoint IDs, final status, endpoint/model, total/node/tool durations, RAG hits/citations, cycle interventions, warnings, and categorized errors. Values may be zero but fields may not be absent.

Inject a saver whose `put` raises `OSError("checkpoint write")`; map it to `checkpoint_write_error`, leave `checkpoint_id` unset, and assert the output never says `success` or `saved`.

- [ ] **Step 9: Run focused tests**

Run: `pytest tests/langgraph_agent/test_checkpoint_resume.py tests/langgraph_agent/test_graph_runtime.py -q`

Expected: PASS.

- [ ] **Step 10: Commit Task 7**

```powershell
git add aerospace_agent/langgraph_agent/checkpointer.py aerospace_agent/langgraph_agent/agent.py aerospace_agent/langgraph_agent/__init__.py tests/langgraph_agent/conftest.py tests/langgraph_agent/test_checkpoint_resume.py
git commit -m "feat: persist and resume langgraph conversations"
```

## Chunk 3: Reversible evolution, CLI, and end-to-end verification

### Task 8: Reversible self-evolution transaction backend

**Files:**
- Create: `aerospace_agent/langgraph_agent/services/evolution_policy.py`
- Create: `aerospace_agent/langgraph_agent/services/evolution_store.py`
- Create: `aerospace_agent/langgraph_agent/services/evolution_validators.py`
- Create: `aerospace_agent/langgraph_agent/services/evolution.py`
- Rewrite: `aerospace_agent/langgraph_agent/evolution.py` as compatibility exports
- Modify: `aerospace_agent/langgraph_agent/agent.py`
- Modify: `tests/langgraph_agent/conftest.py`
- Create: `tests/langgraph_agent/test_evolution_service.py`
- Create: `scripts/run_langgraph_acceptance.py`

- [ ] **Step 1: Write failing commit, auto-restore, rollback, conflict, and escape tests**

```python
import json
import os
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.schema import (
    EvolutionFileChange, EvolutionProposal, ValidationResult,
)
from aerospace_agent.langgraph_agent.services.evolution import EvolutionService


def proposal(path, content, operation="update"):
    return EvolutionProposal(
        thread_id="t1", run_id="r1", rationale="test",
        changes=[EvolutionFileChange(operation=operation, path=path, content=content)],
    )


def test_commit_and_manual_rollback_restore_bytes_permissions_and_indexes(workspace, knowledge_service):
    target = workspace / "knowledge/orbital-dynamics/two-body-orbital-dynamics.md"
    before = target.read_bytes()
    before_mode = target.stat().st_mode
    service = EvolutionService(workspace, knowledge_service=knowledge_service)
    changed = "# Two-body orbital dynamics\n\nunique-evolution-term\n"
    record = service.apply(proposal(target.relative_to(workspace), changed))
    assert record.status == "committed"
    assert knowledge_service.search("unique-evolution-term", top_k=1)
    assert record.state_history == [
        "proposed", "staged", "backed_up", "validating", "committed"
    ]
    assert record.manifest[0].before_sha256 and record.manifest[0].after_sha256
    assert record.manifest[0].mode == before_mode
    assert record.proposal_path.is_file()
    assert record.report_path.is_file()
    assert record.manifest_path.is_file()
    assert json.loads(record.manifest_path.read_text(encoding="utf-8"))["files"]
    rolled = service.rollback(record.evolution_id)
    assert rolled.status == "rolled_back"
    assert target.read_bytes() == before
    assert target.stat().st_mode == before_mode
    assert record.evolution_id in (workspace / "knowledge/log.md").read_text(encoding="utf-8")


def test_validation_failure_restores_files(workspace, knowledge_service):
    def fail(_context):
        return ValidationResult(name="forced", passed=False, details="forced failure")
    service = EvolutionService(workspace, validators=[fail], knowledge_service=knowledge_service)
    target = workspace / "knowledge/index.md"
    before = target.read_bytes()
    record = service.apply(proposal(Path("knowledge/index.md"), "broken"))
    assert record.status == "rolled_back"
    assert target.read_bytes() == before
    assert record.state_history[-2:] == ["validation_failed", "rolled_back"]


@pytest.mark.parametrize("fail_at", [2, 3])
def test_commit_failure_restores_create_update_and_delete(workspace, knowledge_service, fail_at):
    target = workspace / "knowledge/index.md"
    deleted = workspace / "knowledge/orbital-dynamics/propagation-validation.md"
    before_target, before_deleted = target.read_bytes(), deleted.read_bytes()
    changes = [
        EvolutionFileChange(operation="create", path=Path("memory/new.md"), content="# new\n"),
        EvolutionFileChange(operation="update", path=Path("knowledge/index.md"), content="# changed\n"),
        EvolutionFileChange(operation="delete", path=deleted.relative_to(workspace)),
    ]
    calls = 0
    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("forced commit failure")
        os.replace(source, destination)
    service = EvolutionService(
        workspace, knowledge_service=knowledge_service, replace_fn=fail_second_replace,
    )
    record = service.apply(EvolutionProposal(thread_id="t", run_id="r", rationale="x", changes=changes))
    assert record.status == "rolled_back"
    assert not (workspace / "memory/new.md").exists()
    assert target.read_bytes() == before_target
    assert deleted.read_bytes() == before_deleted
    assert record.state_history[-2:] == ["commit_failed", "rolled_back"]


@pytest.mark.parametrize("failure_point", ["wiki_log", "affected_test", "rag_rebuild", "graph_rebuild"])
def test_post_write_failure_restores_exact_snapshot(workspace, knowledge_service, failure_point):
    target = workspace / "knowledge/index.md"
    before = target.read_bytes()
    service = EvolutionService(
        workspace, knowledge_service=knowledge_service,
        failure_injection={failure_point: RuntimeError(failure_point)},
    )
    record = service.apply(proposal(Path("knowledge/index.md"), "# changed\n"))
    assert record.status == "rolled_back"
    assert target.read_bytes() == before
    assert record.state_history[-2:] == ["commit_failed", "rolled_back"]


def test_hash_conflict_refuses_overwrite(workspace, knowledge_service):
    service = EvolutionService(workspace, knowledge_service=knowledge_service)
    target = workspace / "knowledge/index.md"
    record = service.apply(proposal(Path("knowledge/index.md"), "# evolved\n"))
    target.write_text("# later human edit\n", encoding="utf-8")
    rolled = service.rollback(record.evolution_id)
    assert rolled.status == "conflict"
    assert rolled.state_history[-2:] == ["rollback_requested", "conflict"]
    assert target.read_text(encoding="utf-8") == "# later human edit\n"


def test_manual_rollback_restores_create_update_delete_and_absence(workspace, knowledge_service):
    target = workspace / "knowledge/index.md"
    deleted = workspace / "knowledge/orbital-dynamics/propagation-validation.md"
    before_target, before_deleted = target.read_bytes(), deleted.read_bytes()
    record = EvolutionService(workspace, knowledge_service=knowledge_service).apply(
        EvolutionProposal(thread_id="t", run_id="r", rationale="mixed", changes=[
            EvolutionFileChange(operation="create", path=Path("memory/created.md"), content="# created\n"),
            EvolutionFileChange(operation="update", path=Path("knowledge/index.md"), content="# changed\n"),
            EvolutionFileChange(operation="delete", path=Path("knowledge/orbital-dynamics/propagation-validation.md")),
        ])
    )
    assert record.status == "committed"
    rolled = EvolutionService(workspace, knowledge_service=knowledge_service).rollback(record.evolution_id)
    assert rolled.status == "rolled_back"
    assert not (workspace / "memory/created.md").exists()
    assert target.read_bytes() == before_target
    assert deleted.read_bytes() == before_deleted


def test_path_traversal_builtin_and_symlink_escape_are_rejected(workspace, knowledge_service):
    service = EvolutionService(workspace, knowledge_service=knowledge_service)
    with pytest.raises(ValueError, match="allowed"):
        service.apply(proposal(Path("aerospace_agent/core/agent.py"), "x"))
    with pytest.raises(ValueError, match="allowed"):
        service.policy.resolve(Path("knowledge/../../outside.md"))
    outside = workspace.parent / "outside-evolution"
    outside.mkdir(exist_ok=True)
    link = workspace / "knowledge/link-out"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="allowed"):
        service.policy.resolve(Path("knowledge/link-out/escape.md"))


def test_evolve_due_applies_enabled_idle_turn_and_capacity_policy(workspace, knowledge_service):
    conversations = FakeConversationSource([
        thread("eligible-idle", idle_minutes=20, turns=6, context_ratio=0.2),
        thread("eligible-capacity", idle_minutes=20, turns=2, context_ratio=0.9),
        thread("active", idle_minutes=1, turns=20, context_ratio=0.9),
    ])
    service = EvolutionService(
        workspace, knowledge_service=knowledge_service,
        conversation_source=conversations, enabled=True, idle_minutes=10,
        min_turns=6, context_capacity_ratio=0.8,
    )
    assert service.due_thread_ids() == ["eligible-capacity", "eligible-idle"]
    service.enabled = False
    assert service.due_thread_ids() == []


def test_evolve_strict_parse_is_noop_and_due_is_at_most_once(workspace, knowledge_service):
    conversations = FakeConversationSource([
        thread("eligible-idle", idle_minutes=20, turns=6, context_ratio=0.2),
    ])
    service = EvolutionService(
        workspace, knowledge_service=knowledge_service, conversation_source=conversations,
        llm=FakeLLM(responses=["not-json", valid_proposal_json()]),
        enabled=True, idle_minutes=10, min_turns=6,
    )
    failed = service.evolve("eligible-idle")
    assert failed.status == "no_op"
    assert not list((workspace / "knowledge").glob("**/*changed*"))
    first = service.evolve_due()
    second = service.evolve_due()
    assert first.changed == 1
    assert second.changed == 0
    assert conversations.last_evolution_checkpoint("eligible-idle") == first.records[0].checkpoint_id


def test_unfinished_external_action_is_recorded_not_executed(workspace, knowledge_service):
    service = EvolutionService(workspace, knowledge_service=knowledge_service, llm=FakeLLM(responses=[valid_proposal_with_external_item_json()]))
    record = service.evolve("eligible-idle")
    assert record.status in {"committed", "no_op"}
    pending = workspace / "memory/pending.md"
    assert pending.is_file()
    assert "external" in pending.read_text(encoding="utf-8").lower()
```

Define `FakeConversationSource` and `thread()` in this test file as small dataclass-backed helpers with `thread_id`, `last_activity`, `turn_count`, `context_ratio`, and `last_evolution_checkpoint`. Initialize `knowledge_service` through the existing fixture before every test.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/langgraph_agent/test_evolution_service.py -q`

Expected: FAIL because the transactional service does not exist.

- [ ] **Step 3: Implement writable-root and eligibility policy in `evolution_policy.py`**

Resolve every target and confirm it is below one of `knowledge/`, `memory/`, `evolved_skills/`, or `workflows/evolved/`. Reject absolute, `..`, resolved symlink escapes, built-in source, and project-external paths. Implement due eligibility: enabled, idle threshold, and `(min_turns or context_capacity_ratio)` with no review after the same checkpoint.

- [ ] **Step 4: Implement evolution manifests and snapshots in `evolution_store.py`**

Create `data/langgraph/evolution/<id>/{staging,backup}` plus `proposal.json`, `manifest.json`, and `report.json`. For every target record operation, prior existence, relative path, exact bytes, mode, before SHA256, and later after SHA256. Journal every state transition before the associated action; load records by evolution ID. Persist the validated proposal and `manifest.json` before staging; rollback must consume that manifest rather than in-memory state; persist the validation/commit report after every terminal transition.

- [ ] **Step 5: Implement validators in `evolution_validators.py`**

Validate Pydantic proposal shape, Markdown links, `knowledge/index.md` references, discovered `SKILL.md` manifests, workflow YAML, and configured affected pytest commands. Each validator returns `ValidationResult`; any failure blocks commit.

- [ ] **Step 6: Implement staging and backup orchestration**

Render create/update content under `staging/`, journal `staged`, snapshot every existing target plus absence markers for creates, then journal `backed_up` and `validating`. No formal file changes occur before all validators pass.

- [ ] **Step 7: Implement compensating commit and failure restore**

Apply create/update/delete in manifest order and preserve prior permissions on updates. If any replace, delete, post-write Wiki index/log append, affected test, or RAG/graph rebuild fails, journal `commit_failed`, restore all existence/bytes/modes, rebuild the prior knowledge state, and journal `rolled_back`.

- [ ] **Step 8: Implement committed rollback and hash conflict**

After successful Wiki index/log/RAG/graph rebuild, record after hashes and `committed`. Manual rollback first journals `rollback_requested`, then compares every current hash/existence with the recorded after manifest; any mismatch journals `conflict` without changing files. Successful rollback restores bytes/modes/existence, rebuilds knowledge, appends the evolution ID to the log, and journals `rolled_back`.

- [ ] **Step 9: Add conversation review and due-scan entry points**

`agent.evolve(thread_id)` loads an immutable checkpoint snapshot and asks the configured LLM for `EvolutionProposal` JSON. Parse/validate strictly; if parsing fails, record a failed/no-op review and make no file changes. Unfinished external actions are recorded under `memory/pending.md`, not executed.

`agent.evolve_due()` obtains thread summaries from the checkpointer adapter, filters with `EvolutionPolicy`, sorts thread IDs, and evolves each eligible snapshot at most once. Return records for changed/failed reviews and a count of no-op reviews.

- [ ] **Step 10: Run tests**

Run: `pytest tests/langgraph_agent/test_evolution_service.py tests/test_skill_registry_framework.py -q`

Expected: PASS.

- [ ] **Step 11: Commit Task 8**

```powershell
git add aerospace_agent/langgraph_agent/services/evolution_policy.py aerospace_agent/langgraph_agent/services/evolution_store.py aerospace_agent/langgraph_agent/services/evolution_validators.py aerospace_agent/langgraph_agent/services/evolution.py aerospace_agent/langgraph_agent/evolution.py aerospace_agent/langgraph_agent/agent.py tests/langgraph_agent/conftest.py tests/langgraph_agent/test_evolution_service.py scripts/run_langgraph_acceptance.py
git commit -m "feat: add reversible agent evolution"
```

### Task 9: CLI, JSON Schema artifacts, and operator documentation

**Files:**
- Rewrite: `start_langgraph_agent.py`
- Create: JSON files under `schemas/langgraph_agent/`
- Create: `docs/LANGGRAPH_AGENT.md`
- Create: `tests/langgraph_agent/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

```python
import json
from pathlib import Path
import subprocess
import sys


def run_cli(workspace: Path, *args):
    return subprocess.run(
        [
            sys.executable, "start_langgraph_agent.py",
            "--workspace", str(workspace), *args,
        ],
        text=True, capture_output=True, timeout=60,
    )


def test_init_status_graph_and_mock_task_commands(tmp_path):
    init = run_cli(tmp_path, "--init-knowledge", "--json")
    assert init.returncode == 0
    assert json.loads(init.stdout)["created"] == 6

    status = run_cli(tmp_path, "--knowledge-status", "--json")
    assert status.returncode == 0
    assert json.loads(status.stdout)["wiki_pages"] == 6

    graph_path = tmp_path / "reports/graph.html"
    graph = run_cli(tmp_path, "--knowledge-graph", str(graph_path), "--json")
    graph_payload = json.loads(graph.stdout)
    assert graph.returncode == 0
    assert Path(graph_payload["html_path"]).is_file()
    assert Path(graph_payload["json_path"]).is_file()

    task = run_cli(tmp_path, "--mock", "--task", "What is two-body dynamics?", "--thread", "cli", "--json")
    payload = json.loads(task.stdout)
    assert task.returncode == 0
    assert payload["status"] == "success"
    assert payload["citations"]

    history = run_cli(tmp_path, "--checkpoint-history", "cli", "--json")
    assert history.returncode == 0
    assert json.loads(history.stdout)["checkpoints"]

    due = run_cli(tmp_path, "--mock", "--evolve-due", "--json")
    assert due.returncode == 0
    assert {"eligible", "changed", "failed", "no_op"} <= json.loads(due.stdout)

    evolve = run_cli(tmp_path, "--mock", "--evolve", "cli", "--json")
    assert evolve.returncode == 0
    assert json.loads(evolve.stdout)["thread_id"] == "cli"

    seeded_record = seed_deterministic_evolution(tmp_path)
    rollback = run_cli(tmp_path, "--rollback", seeded_record.evolution_id, "--json")
    assert rollback.returncode == 0
    assert json.loads(rollback.stdout)["status"] == "rolled_back"

    schemas = run_cli(tmp_path, "--export-schemas", "--json")
    assert schemas.returncode == 0
    schema_payload = json.loads(schemas.stdout)
    assert len(schema_payload["files"]) == 9
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in schema_payload["files"])

    missing = run_cli(tmp_path, "--rollback", "missing-id", "--json")
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["error_code"] == "EVOLUTION_NOT_FOUND"


def test_json_mode_has_one_document_and_diagnostics_only_on_stderr(tmp_path):
    result = run_cli(tmp_path, "--init-knowledge", "--json")
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(result.stdout)
    assert result.stdout[end:].strip() == ""
    assert payload["created"] == 6
```

Add a scheduler lifecycle test with a fake clock and fake Agent: start REPL scheduler, advance past idle threshold, assert one `evolve_due()` call, call `stop()`, and assert its thread is no longer alive. It must not be constructed by the one-shot `--task` branch. Add explicit `--repl`, `--stream`, and `--tools` parser/dispatch tests using a fake Agent; assert each returns one JSON document in `--json` mode.

Define `seed_deterministic_evolution(tmp_path)` in the test module: initialize the six-page Wiki, construct an `EvolutionService` with a fixed `EvolutionProposal` that updates one page, apply it, and return its committed record. This makes the successful CLI rollback test independent of model availability.

- [ ] **Step 2: Run tests and verify missing commands fail**

Run: `pytest tests/langgraph_agent/test_cli.py -q`

Expected: FAIL because the current CLI does not implement these arguments or clean JSON output.

- [ ] **Step 3: Implement global workspace/config/json options and mutually exclusive actions**

Add `--workspace`, `--config`, `--json`, `--mock`, and `--thread`. Action options are mutually exclusive: repl/task/stream/tools/init/status/graph/history/evolve/evolve-due/rollback/export-schemas. Load settings once. Do not auto-spawn a visible console. If Qwen is unavailable, exit 2 with `LLM_UNAVAILABLE` unless `--mock` is explicit. `--json` emits exactly one JSON document; diagnostics go to stderr.

- [ ] **Step 4: Implement knowledge, checkpoint, and evolution CLI adapters**

Return the exact tested fields and exit codes. `--rollback` maps not-found to exit 2. All paths printed in JSON are absolute. CLI functions delegate to Agent/service APIs and do not query SQLite tables or edit Wiki files directly.

- [ ] **Step 5: Add a bounded REPL evolution scheduler**

In REPL mode, a daemon thread may call `evolve_due()` only after configured idle/min-turn conditions. It must stop on Agent close and never keep one-shot `--task` alive. `--evolve-due` provides the scheduler-friendly one-shot path.

- [ ] **Step 6: Export exact JSON Schema artifacts from the Pydantic source of truth**

Add `--export-schemas`; generate deterministic UTF-8 JSON with sorted keys and remove `.gitkeep`. Map models to exact filenames:

```text
AgentInput -> agent-input-v1.json
AgentOutput -> agent-output-v1.json
Decision -> decision-v1.json
EvidenceItem -> evidence-item-v1.json
ToolCallRequest -> tool-call-request-v1.json
ToolCallResponse -> tool-call-response-v1.json
EvolutionProposal -> evolution-proposal-v1.json
EvolutionRecord -> evolution-record-v1.json
ValidationResult -> validation-result-v1.json
```

Test the directory contains exactly these nine files and every parsed artifact equals the corresponding normalized `model_json_schema()`.

- [ ] **Step 7: Write operator documentation**

Document architecture boundaries, directory ownership, install, Qwen endpoint, all CLI commands, MCP stdio behavior, checkpoint semantics, evolution safeguards, rollback conflict, and the exact scope of the six-seed knowledge base.

- [ ] **Step 8: Run CLI and documentation tests**

Run: `pytest tests/langgraph_agent/test_cli.py tests/langgraph_agent/test_schema_state.py -q`

Expected: PASS.

- [ ] **Step 9: Commit Task 9**

```powershell
git add start_langgraph_agent.py schemas/langgraph_agent docs/LANGGRAPH_AGENT.md tests/langgraph_agent/test_cli.py
git commit -m "feat: expose langgraph agent operations"
```

### Task 10: Local Qwen acceptance, regression, and evidence report

**Files:**
- Create: `tests/langgraph_agent/test_qwen_acceptance.py`
- Modify: `scripts/run_langgraph_acceptance.py` (created in Task 8; Task 10 adds the test/report subcommands)
- Create: `reports/langgraph_agent_acceptance_2026-07-10.json`
- Create: `reports/langgraph_agent_acceptance_2026-07-10.md`
- Modify only for verified defects: files from Tasks 1–9

- [ ] **Step 1: Write opt-in real-model acceptance tests**

```python
import pytest


@pytest.mark.qwen3
@pytest.mark.integration
def test_qwen_endpoint_models_and_direct_completion(qwen_client):
    models = qwen_client.models()
    assert "qwythos" in models
    response = qwen_client.chat(
        "State the governing acceleration for two-body orbital dynamics.",
        max_tokens=128,
    )
    assert response.content
    assert qwen_client.claim_support(response.content, ["The governing acceleration is -mu r / |r|^3."])["unsupported_claims"] == []


@pytest.mark.qwen3
@pytest.mark.integration
def test_qwen_agent_answers_from_seed_evidence(qwen_agent):
    first = qwen_agent.run("What is two-body orbital dynamics?", thread_id="qwen-accept")
    assert first.status == "success"
    assert any("two-body-orbital-dynamics.md" in c.page_path for c in first.citations)
    support = qwen_agent.verify_answer_against_citations(first.answer, first.citations)
    assert support["unsupported_claims"] == []
    assert support["claims"]

    second = qwen_agent.run("What assumptions did you just mention?", thread_id="qwen-accept")
    snapshot = qwen_agent.get_conversation_state("qwen-accept")
    assert len(snapshot.values["messages"]) >= 4
    assert second.checkpoint_id
```

The fixture first verifies `/v1/models` contains `qwythos`, then uses the configured local endpoint. No test starts or stops the user's model process. If the endpoint is unavailable, the fixture writes `qwen_status=unavailable` to the evidence collector and marks the real-model acceptance as `blocked` (not passed); offline tests still run.

- [ ] **Step 2: Run all offline LangGraph tests**

Run: `pytest tests/langgraph_agent -q -m "not qwen3"`

Expected: PASS.

- [ ] **Step 3: Run the standard MCP protocol smoke independently**

Run: `pytest tests/langgraph_agent/test_mcp_gateway.py -q -m integration`

Expected: initialize/list/call PASS; captured stdout has no protocol-external startup lines.

- [ ] **Step 4: Run the real local Qwen test**

Run: `pytest tests/langgraph_agent/test_qwen_acceptance.py -q -m qwen3`

Expected: direct completion and cited two-turn Agent tests PASS when the endpoint is available. If unavailable, the command reports `blocked` with the connection error and exits non-zero; this is an external limitation, not a passed acceptance. Record endpoint model ID, elapsed time, steps, citations, and checkpoint ID. A non-empty answer alone is not sufficient.

- [ ] **Step 5: Run related regressions**

Run: `pytest tests/test_context_framework.py tests/test_mcp_schema_framework.py tests/test_rag_multisource.py tests/test_skill_registry_framework.py aerospace_agent/mcp/tests -q`

Expected: PASS or explicit pre-existing/environment-specific skip with reason.

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q -m "not qwen3"`

Expected: PASS for the complete offline suite. The opt-in Qwen suite runs separately in Step 4; if unavailable it is reported `blocked`, not silently collected or treated as passed. If unrelated pre-existing tests fail, prove the failure on the design commit or isolate it with evidence; do not silently exclude it.

- [ ] **Step 7: Exercise the documented terminal flow in an isolated workspace**

```powershell
$acceptance = Join-Path $env:TEMP ("aerospace-agent-acceptance-" + [guid]::NewGuid())
python start_langgraph_agent.py --workspace $acceptance --init-knowledge --json
python start_langgraph_agent.py --workspace $acceptance --knowledge-status --json
python start_langgraph_agent.py --workspace $acceptance --knowledge-graph (Join-Path $acceptance "reports/graph.html") --json
python start_langgraph_agent.py --workspace $acceptance --mock --task "What is two-body orbital dynamics?" --thread acceptance-qwen --json
python start_langgraph_agent.py --workspace $acceptance --mock --task "What assumptions did you mention?" --thread acceptance-qwen --json
python start_langgraph_agent.py --workspace $acceptance --checkpoint-history acceptance-qwen --json
```

Expected: every command exits 0 with schema-valid JSON, generated paths exist under `$acceptance`, and the repository's `data/` remains unchanged.

- [ ] **Step 8: Run a real evolution commit/rollback acceptance in an isolated workspace**

Run: `python scripts/run_langgraph_acceptance.py --workspace $acceptance --evolution-roundtrip --json`

Expected: exit 0, JSON contains `{\"evolution_status\": \"rolled_back\"}`, and before/after/rollback SHA256 values prove exact restoration. Do not mutate the repository's tracked Wiki during this test.

- [ ] **Step 9: Generate evidence reports**

Run: `python scripts/run_langgraph_acceptance.py --workspace $acceptance --run-tests --qwen --output-json reports/langgraph_agent_acceptance_2026-07-10.json --output-markdown reports/langgraph_agent_acceptance_2026-07-10.md`

The JSON report contains command, exit code, duration, Python/package versions, test counts, skips, Qwen model metadata/status, run/checkpoint IDs, citations, evolution ID, and before/after/rollback hashes. The Markdown report summarizes only evidence present in JSON and lists unverified semantic claims separately. When Qwen is unavailable it records `blocked`, never `passed`.

- [ ] **Step 10: Invoke verification-before-completion and inspect the final diff**

Run: `git diff --check` and `git status --short`.

Confirm no generated checkpoint DB, RAG binary, evolution backup, `.superpowers/`, Qwen output, or unrelated user file is staged.

- [ ] **Step 11: Request code review and fix blocking findings**

Use `@superpowers:requesting-code-review` against the approved spec and this plan. Re-run affected tests after every fix.

- [ ] **Step 12: Commit verified acceptance artifacts and any tested fixes**

```powershell
git add tests/langgraph_agent/test_qwen_acceptance.py scripts/run_langgraph_acceptance.py reports/langgraph_agent_acceptance_2026-07-10.json reports/langgraph_agent_acceptance_2026-07-10.md
# If Tasks 1–9 were fixed during verification, stage those exact changed files here too.
git diff --cached --check
git commit -m "test: verify langgraph aerospace agent"
```

## Final completion criteria

- [ ] All spec requirements map to at least one passing test or an explicitly reported external limitation.
- [ ] The six-seed Wiki, index, log, graph JSON/HTML, and citations are generated and internally consistent.
- [ ] SQLite continuation survives a new Agent instance; interrupted state resumes from `snapshot.next`.
- [ ] Loop and recursion protections terminate deliberately constructed non-terminating paths.
- [ ] MCP stdio initialize/list/call passes with protocol-clean stdout.
- [ ] Evolution commit, automatic restore, manual rollback, hash conflict, and path escape tests pass.
- [ ] Local `qwythos` completes a cited knowledge answer and a same-thread follow-up.
- [ ] Full test results and every skip/failure are captured in the acceptance report.
- [ ] No claim of book ingestion, model answer correctness, or skill improvement is made without supporting evidence.
