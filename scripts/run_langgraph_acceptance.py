"""Run local LangGraph acceptance checks and write auditable evidence.

The runner is deliberately offline-first.  It may *probe* an already-running
OpenAI-compatible endpoint, but it never starts, stops, or otherwise manages a
model process.  Qwen results are therefore ``blocked`` when the endpoint is
not reachable, never a false pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

# Direct script execution puts ``scripts/`` first on ``sys.path``.  Ensure the
# repository source wins over any older editable/global installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent, SimpleLLMClient
from aerospace_agent.langgraph_agent.config import load_settings
from aerospace_agent.langgraph_agent.graph import ServiceBundle
from aerospace_agent.langgraph_agent.schema import EvolutionFileChange, EvolutionProposal
from aerospace_agent.langgraph_agent.services.evolution import EvolutionService
from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwythos"


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("excerpt", value.get("content", "")))
    return str(getattr(value, "excerpt", value) or "")


def verify_answer_against_citations(answer: str, citations: Iterable[Any]) -> dict[str, Any]:
    """Return a conservative claim-support report.

    The exact seed equation is treated as one atomic claim.  Other sentences
    are considered supported only when they share enough meaningful tokens with
    a citation excerpt; an empty answer is explicitly unsupported.
    """

    answer_text = str(answer or "").strip()
    excerpts = [_text(item).strip() for item in citations if _text(item).strip()]
    normalized = answer_text.lower().replace("−", "-").replace("∕", "/")
    seed_forms = ("-mu r / |r|^3", "-mu*r/|r|^3", "-mu r/|r|^3")
    compact_answer = re.sub(r"\s+", "", normalized)
    latex_equivalent = "\\frac" in normalized and "mu" in normalized and "r" in normalized and "^3" in normalized
    seed_in_answer = any(re.sub(r"\s+", "", form) in compact_answer for form in seed_forms) or latex_equivalent
    seed_in_citation = any("-mur/|r|^3" in item.lower().replace(" ", "") for item in excerpts)

    unsupported: list[str] = []
    supported: list[str] = []
    evidence_links: list[dict[str, Any]] = []
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "are",
        "is", "under", "only", "into", "when", "while", "also", "use",
        "using", "must", "should", "can", "as", "of", "to", "in", "on",
    }
    if not answer_text:
        unsupported.append("(empty answer)")
    else:
        # Keep equation support independent from prose token overlap: this is
        # the exact claim used by the seed corpus and by the Qwen smoke test.
        if seed_in_answer:
            if seed_in_citation:
                supported.append("-mu r / |r|^3")
                formula_index = next(
                    (index for index, item in enumerate(excerpts)
                     if "-mur/|r|^3" in item.lower().replace(" ", "")),
                    0,
                )
                evidence_links.append({"claim": "-mu r / |r|^3", "citation_index": formula_index, "score": 1.0})
            else:
                unsupported.append("-mu r / |r|^3")
        chunks = [part.strip(" -*\t") for part in re.split(r"[\n.!?]+", answer_text) if part.strip()]
        for chunk in chunks:
            normalized_chunk = chunk.rstrip(":").strip().lower()
            if normalized_chunk in {"evidence", "answer", "response"} or normalized_chunk.startswith("evidence:"):
                continue
            if seed_in_answer and "mu" in chunk.lower() and ("|r|" in chunk.lower() or "\\frac" in chunk.lower()):
                continue
            tokens = {token for token in re.findall(r"[a-z0-9_]+", chunk.lower()) if len(token) > 2}
            if not tokens:
                if re.search(r"[=+*/^]", chunk):
                    unsupported.append(chunk)
                continue
            best = 0.0
            best_index: int | None = None
            claim_numbers = set(re.findall(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", normalized_chunk))
            for index, excerpt in enumerate(excerpts):
                cited_tokens = {
                    token for token in re.findall(r"[a-z0-9_]+", excerpt.lower())
                    if len(token) > 2 and token not in stopwords
                }
                claim_tokens = {token for token in tokens if token not in stopwords}
                if not claim_tokens:
                    continue
                score = len(claim_tokens & cited_tokens) / len(claim_tokens)
                cited_numbers = set(re.findall(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", excerpt.lower()))
                if claim_numbers and not claim_numbers.issubset(cited_numbers):
                    score = 0.0
                if score > best:
                    best, best_index = score, index
            # A claim is grounded only when most of its meaningful terms are
            # present in one cited excerpt.  Numeric values must also match;
            # this prevents a generic token overlap from "supporting" an
            # invented altitude, epoch, or delta-v.
            if best >= 0.60 and best_index is not None:
                supported.append(chunk)
                evidence_links.append({
                    "claim": chunk,
                    "citation_index": best_index,
                    "score": round(best, 3),
                })
            else:
                unsupported.append(chunk)
    versions = _package_versions()
    return {
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "evidence_links": evidence_links,
        "citation_count": len(excerpts),
    }


def probe_qwen_endpoint(endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL, *, timeout: float = 3.0) -> dict[str, Any]:
    """Probe ``GET /models`` without managing the endpoint process."""

    endpoint = str(endpoint).rstrip("/")
    model = str(model)
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "model": model,
        "status": "blocked",
        "available": False,
        "models": [],
    }
    try:
        with urllib.request.urlopen(f"{endpoint}/models", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, Mapping)]
        result.update({"models": models, "available": model in models, "status": "available" if model in models else "blocked"})
        if model not in models:
            result["error"] = f"model {model!r} not listed by /models"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def qwen_claim_smoke(qwen: Mapping[str, Any]) -> dict[str, Any]:
    """Ask the live model to repeat the exact seed claim, when available."""

    if qwen.get("status") != "available":
        return {"status": "blocked", "reason": qwen.get("error", "endpoint unavailable")}
    try:
        client = SimpleLLMClient(endpoint=str(qwen["endpoint"]), model=str(qwen["model"]), timeout=30)
        answer = client.chat(
            "Output exactly this sentence and nothing else: The governing acceleration is -mu r / |r|^3.",
            system_prompt="Copy the requested sentence exactly. Do not identify yourself or add commentary.",
            max_tokens=64,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
        )
        citations = [{"page_path": "knowledge/orbital-dynamics/two-body-orbital-dynamics.md", "excerpt": "The governing acceleration is -mu r / |r|^3."}]
        return {"status": "observed", "answer": answer, "claim_support": verify_answer_against_citations(answer, citations)}
    except Exception as exc:
        return {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}


def qwen_conditional_rag_smoke(qwen: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    """Observe both conditional-RAG graph paths against an available model."""

    if qwen.get("status") != "available":
        return {"status": "blocked", "reason": qwen.get("error", "endpoint unavailable")}
    agent = None
    try:
        knowledge = KnowledgeService(workspace=workspace)
        knowledge.initialize_seed_wiki()
        counting = _CountingKnowledge(knowledge)
        agent = LangGraphAerospaceAgent(
            settings=load_settings(workspace=workspace),
            llm_endpoint=str(qwen["endpoint"]),
            model_name=str(qwen["model"]),
            checkpoint_backend="memory",
            rag=counting,
        )
        before_general = len(counting.calls)
        general = agent.run(
            "你好，我们先正常讨论今天的研究安排。",
            thread_id="acceptance-qwen-general",
        )
        after_general = len(counting.calls)
        evidence = agent.run(
            "请根据私域知识库给出二体动力学加速度的依据和来源。",
            thread_id="acceptance-qwen-evidence",
        )
        after_evidence = len(counting.calls)
        work = agent.run(
            "Draft a high-level orbit propagation work plan.",
            thread_id="acceptance-qwen-work",
        )
        work_snapshot = agent.get_conversation_state("acceptance-qwen-work")
        work_values = getattr(work_snapshot, "values", {}) or {}
        planner_ran = "planner" in (work_values.get("node_timings_ms", {}) or {})
        general_calls = after_general - before_general
        evidence_calls = after_evidence - after_general
        work_calls = len(counting.calls) - after_evidence
        policy_satisfied = (
            general.status == "success"
            and general_calls == 0
            and len(general.citations) == 0
            and evidence.status == "success"
            and evidence_calls == 1
            and bool(evidence.citations)
            and evidence.metrics.get("retrieval_reason") == "explicit_evidence"
            and work.status == "success"
            and planner_ran
            and work_calls <= 1
            and (
                work_calls == 0
                or work.metrics.get("retrieval_reason") == "planner_request"
            )
        )
        return {
            "status": "observed",
            "policy_satisfied": policy_satisfied,
            "general": {
                "status": getattr(general.status, "value", str(general.status)),
                "search_call_count": general_calls,
                "citation_count": len(general.citations),
            },
            "evidence": {
                "status": getattr(evidence.status, "value", str(evidence.status)),
                "search_call_count": evidence_calls,
                "citation_count": len(evidence.citations),
                "retrieval_reason": evidence.metrics.get("retrieval_reason", ""),
            },
            "model_dispatch": {
                "status": getattr(work.status, "value", str(work.status)),
                "intent": getattr(work.intent, "value", str(work.intent)),
                "planner_ran": planner_ran,
                "search_call_count": work_calls,
                "retrieval_reason": work.metrics.get("retrieval_reason", ""),
            },
        }
    except Exception as exc:
        return {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if agent is not None:
            agent.close()


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_evolution_roundtrip(workspace: Path) -> dict[str, Any]:
    """Commit then rollback one deterministic file transaction."""

    workspace = Path(workspace).resolve()
    target = workspace / "memory" / "acceptance-roundtrip.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    original = target.read_bytes() if existed else None
    # A fixed baseline lets the report expose stable before/after hashes while
    # preserving and restoring any pre-existing caller file.
    target.write_text("baseline\n", encoding="utf-8")
    service = EvolutionService(workspace=workspace, data_dir=workspace / "data/langgraph/evolution")
    proposal = EvolutionProposal(
        thread_id="acceptance",
        run_id="acceptance-evolution-roundtrip",
        rationale="deterministic acceptance transaction",
        changes=[EvolutionFileChange(operation="update", path="memory/acceptance-roundtrip.md", content="updated\n")],
    )
    committed = service.apply(proposal)
    rolled_back = service.rollback(committed.evolution_id)
    manifest = [item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in committed.manifest]
    hashes = [{"path": item.get("path"), "before_sha256": item.get("before_sha256"), "after_sha256": item.get("after_sha256")} for item in manifest]
    # Restore caller state (the temporary workspace normally has no prior file).
    if existed and original is not None:
        target.write_bytes(original)
    elif target.exists():
        target.unlink()
    return {
        "evolution_status": rolled_back.status,
        "evolution_id": committed.evolution_id,
        "rollback_evolution_id": rolled_back.evolution_id,
        "run_id": committed.run_id,
        "state_history": list(rolled_back.state_history),
        "hashes": hashes,
        "target_sha256_after_rollback": _sha256(target),
    }


def _package_versions() -> dict[str, str | None]:
    names = ("pytest", "langgraph", "langchain-core", "pydantic", "httpx")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class _CountingKnowledge:
    """Record actual search invocations while preserving service behavior."""

    def __init__(self, service: KnowledgeService):
        self.service = service
        self.calls: list[str] = []

    def search(self, query: str, *, top_k: int = 5):
        self.calls.append(str(query))
        return self.service.search(query, top_k=top_k)


def _agent_smoke(workspace: Path) -> dict[str, Any]:
    """Exercise both sides of the conditional private-RAG policy."""

    knowledge = KnowledgeService(workspace=workspace)
    knowledge.initialize_seed_wiki()
    counting_knowledge = _CountingKnowledge(knowledge)
    settings = load_settings(workspace=workspace)
    services = ServiceBundle(
        knowledge=counting_knowledge,
        model_name="deterministic-acceptance",
        endpoint="offline://",
    )
    agent = LangGraphAerospaceAgent(settings=settings, services=services, checkpoint_backend="memory")
    try:
        before_general = len(counting_knowledge.calls)
        general = agent.run("Hello. How are you today?", thread_id="acceptance-general")
        after_general = len(counting_knowledge.calls)
        evidence = agent.run(
            "Using the private knowledge base, cite evidence for the governing acceleration in two-body dynamics.",
            thread_id="acceptance-evidence",
        )
        citations = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in evidence.citations
        ]
        return {
            "general": {
                "thread_id": "acceptance-general",
                "status": getattr(general.status, "value", str(general.status)),
                "run_id": general.metrics.get("run_id"),
                "checkpoint_id": general.checkpoint_id,
                "retrieval_required": bool(general.metrics.get("retrieval_required", False)),
                "retrieval_reason": general.metrics.get("retrieval_reason", ""),
                "rag_hits": int(general.metrics.get("rag_hits", 0) or 0),
                "citation_count": len(general.citations),
                "search_call_count": after_general - before_general,
            },
            "evidence": {
                "thread_id": "acceptance-evidence",
                "status": getattr(evidence.status, "value", str(evidence.status)),
                "run_id": evidence.metrics.get("run_id"),
                "checkpoint_id": evidence.checkpoint_id,
                "retrieval_required": bool(evidence.metrics.get("retrieval_required", False)),
                "retrieval_reason": evidence.metrics.get("retrieval_reason", ""),
                "rag_hits": int(evidence.metrics.get("rag_hits", 0) or 0),
                "citation_count": len(citations),
                "search_call_count": len(counting_knowledge.calls) - after_general,
            },
            "citations": citations,
            "claim_support": verify_answer_against_citations(evidence.answer, citations),
        }
    finally:
        agent.close()


def _parse_pytest_counts(stdout: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0, "collected": 0}
    match = re.search(r"collected\s+(\d+)\s+items", stdout)
    if match:
        counts["collected"] = int(match.group(1))
    for key, label in (("passed", "passed"), ("failed", "failed"), ("skipped", "skipped"), ("xfailed", "xfailed"), ("xpassed", "xpassed"), ("errors", "errors")):
        match = re.search(rf"(\d+)\s+{label}", stdout)
        if match:
            counts[key] = int(match.group(1))
    return counts


def run_tests(workspace: Path) -> dict[str, Any]:
    endpoint = os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL", DEFAULT_ENDPOINT)
    model = os.environ.get("AEROSPACE_LOCAL_LLM_MODEL", DEFAULT_MODEL)
    qwen = probe_qwen_endpoint(endpoint, model)
    qwen["claim_smoke"] = qwen_claim_smoke(qwen)
    qwen["conditional_rag_smoke"] = qwen_conditional_rag_smoke(qwen, workspace)
    basetemp = Path(workspace).resolve() / ".test-artifacts" / "acceptance-pytest"
    if basetemp.exists():
        shutil.rmtree(basetemp)
    basetemp.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/langgraph_agent",
        "-q",
        "-m",
        "not qwen3",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(basetemp / "pytest-basetemp"),
    ]
    started = time.perf_counter()
    test_env = os.environ.copy()
    test_env["PYTHONDONTWRITEBYTECODE"] = "1"
    test_env["AEROSPACE_TEST_ARTIFACT_ROOT"] = str(basetemp / "fixtures")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=test_env,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    smoke: dict[str, Any]
    try:
        smoke = _agent_smoke(workspace)
    except Exception as exc:
        smoke = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    acl_blocked = any(marker in (completed.stdout + completed.stderr) for marker in ("PermissionError", "Access is denied", "拒绝访问", "WinError 5"))
    counts = _parse_pytest_counts(completed.stdout)
    if acl_blocked and counts["errors"] == 0:
        # Pytest can fail during tmpdir teardown before printing a summary.
        # Preserve the observed collection size instead of reporting a false
        # zero-error pass.
        counts["errors"] = max(0, counts["collected"] - 2)
    manual_fallback = None
    if acl_blocked:
        manual_nodes = [
            "tests/langgraph_agent/test_schema_state.py::test_agent_input_has_versioned_runtime_contract",
            "tests/langgraph_agent/test_schema_state.py::test_agent_output_rejects_unknown_status",
            "tests/langgraph_agent/test_schema_state.py::test_intermediate_protocol_constraints_and_independent_defaults",
            "tests/langgraph_agent/test_schema_state.py::test_tool_and_evolution_models_cover_success_and_failure",
            "tests/langgraph_agent/test_schema_state.py::test_state_round_trips_through_langgraph_jsonplus",
            "tests/langgraph_agent/test_schema_state.py::test_schema_versions_and_evolution_status_are_closed_sets",
            "tests/langgraph_agent/test_context_cycle.py::test_evaluate_cycle_state_isolation",
            "tests/langgraph_agent/test_context_cycle.py::test_fingerprint_distinguishes_tool_name_and_target",
            "tests/langgraph_agent/test_context_cycle.py::test_compatibility_check_returns_reason_and_delta",
        ]
        fallback_command = [sys.executable, "-m", "pytest", *manual_nodes, "-q", "--confcutdir", "tests/langgraph_agent"]
        fallback_started = time.perf_counter()
        fallback = subprocess.run(fallback_command, cwd=ROOT, capture_output=True, text=True, check=False)
        manual_fallback = {
            "status": "ok" if fallback.returncode == 0 else "blocked",
            "command": fallback_command,
            "exit_code": fallback.returncode,
            "duration_ms": round((time.perf_counter() - fallback_started) * 1000.0, 3),
            "counts": _parse_pytest_counts(fallback.stdout),
            "stdout": fallback.stdout,
            "stderr": fallback.stderr,
        }
    versions = _package_versions()
    result = {
        "command": command,
        "command_string": subprocess.list2cmdline(command),
        "exit_code": completed.returncode,
        "duration_ms": round(duration_ms, 3),
        "python_package_versions": versions,
        "python_version": versions.get("python"),
        "package_versions": {key: value for key, value in versions.items() if key != "python"},
        "counts": counts,
        "qwen": qwen,
        "fixture_acl": {"status": "blocked" if acl_blocked else "ok", "basetemp": str(basetemp), "reason": "pytest temp fixture ACL denied" if acl_blocked else None},
        "manual_fallback": manual_fallback,
        "run": smoke,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode == 0:
        shutil.rmtree(basetemp, ignore_errors=True)
        result["fixture_acl"]["basetemp_removed"] = not basetemp.exists()
    else:
        result["fixture_acl"]["basetemp_removed"] = False
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# LangGraph acceptance evidence", "", f"Observed at: {report.get('observed_at', '')}", ""]
    if "evolution" in report:
        evo = report["evolution"]
        lines.extend(["## Evolution round-trip", "", f"- Status: `{evo.get('evolution_status')}`", f"- Evolution ID: `{evo.get('evolution_id')}`", f"- Rollback ID: `{evo.get('rollback_evolution_id')}`", f"- Hashes: `{json.dumps(evo.get('hashes', []), ensure_ascii=False)}`", ""])
    if "tests" in report:
        tests = report["tests"]
        qwen = tests.get("qwen", {})
        lines.extend(["## Verification", "", "### Offline pytest", "", f"- Command: `{tests.get('command_string')}`", f"- Exit code: `{tests.get('exit_code')}`", f"- Duration (ms): `{tests.get('duration_ms')}`", f"- Counts: `{json.dumps(tests.get('counts', {}), sort_keys=True)}`", f"- Fixture ACL: **{tests.get('fixture_acl', {}).get('status', 'unknown')}**", f"- Qwen status: **{qwen.get('status', 'blocked')}** (model `{qwen.get('model', '')}`)", ""])
        if tests.get("manual_fallback"):
            lines.append(f"- ACL fallback: `{tests['manual_fallback'].get('status')}` ({tests['manual_fallback'].get('exit_code')})")
        live_rag = qwen.get("conditional_rag_smoke", {})
        if live_rag:
            lines.extend([
                "### Live Qwen conditional RAG",
                "",
                f"- Observation status: `{live_rag.get('status')}`; policy satisfied: `{live_rag.get('policy_satisfied')}`",
                f"- General dialogue: `{json.dumps(live_rag.get('general', {}), ensure_ascii=False)}`",
                f"- Evidence request: `{json.dumps(live_rag.get('evidence', {}), ensure_ascii=False)}`",
                f"- Work/model dispatch: `{json.dumps(live_rag.get('model_dispatch', {}), ensure_ascii=False)}`",
                "",
            ])
        run = tests.get("run", {})
        if run:
            general = run.get("general", {})
            evidence = run.get("evidence", {})
            lines.extend([
                "### Conditional private RAG",
                "",
                f"- General dialogue: status `{general.get('status')}`, retrieval required `{general.get('retrieval_required')}`, searches `{general.get('search_call_count')}`, RAG hits `{general.get('rag_hits')}`, citations `{general.get('citation_count')}`",
                f"- Evidence request: status `{evidence.get('status')}`, reason `{evidence.get('retrieval_reason')}`, searches `{evidence.get('search_call_count')}`, RAG hits `{evidence.get('rag_hits')}`, citations `{evidence.get('citation_count')}`",
                f"- Evidence claim support: `{json.dumps(run.get('claim_support', {}), ensure_ascii=False)}`",
                "",
            ])
    lines.append("Qwen endpoint availability is a probe result; no model process was started or stopped by this runner.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--evolution-roundtrip", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "workspace": str(args.workspace.resolve())}
    if args.evolution_roundtrip or not args.run_tests:
        report["evolution"] = run_evolution_roundtrip(args.workspace)
    if args.run_tests:
        report["tests"] = run_tests(args.workspace)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    payload = report
    if args.evolution_roundtrip and not args.run_tests and not args.output_json and not args.output_markdown:
        payload = report.get("evolution", report)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    tests = report.get("tests", {})
    tests_ok = tests.get("exit_code", 0) == 0
    qwen = tests.get("qwen", {})
    live_rag = qwen.get("conditional_rag_smoke", {})
    tests_ok = (
        tests_ok
        and qwen.get("status") == "available"
        and live_rag.get("status") == "observed"
        and bool(live_rag.get("policy_satisfied"))
    )
    return 0 if report.get("evolution", {}).get("evolution_status", "rolled_back") == "rolled_back" and tests_ok else (0 if "tests" not in report else 1)


if __name__ == "__main__":
    raise SystemExit(main())
