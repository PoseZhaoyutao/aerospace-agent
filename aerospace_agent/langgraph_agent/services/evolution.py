"""Reversible, journaled workspace evolution transactions."""
from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..schema import EvolutionFileChange, EvolutionProposal, EvolutionRecord
from .evolution_policy import ALLOWED_ROOT_NAMES, EvolutionPolicy, parse_llm_proposal, validate_target_path
from .evolution_store import EvolutionStore, decode_bytes, encode_bytes, sha256_bytes
from .evolution_validators import ValidationResult, Validator, run_validators


class EvolutionService:
    """Apply create/update/delete proposals with compensating rollback."""

    TERMINAL = {"committed", "rolled_back", "conflict"}
    CORE_MANAGED_ROOTS = ("evolved_skills", "workflows/evolved")

    def __init__(self, workspace: str | os.PathLike[str] | None = None, *, data_dir: str | os.PathLike[str] | None = None,
                 allowed_roots: Iterable[str | os.PathLike[str]] | None = None, policy: EvolutionPolicy | None = None,
                 validators: Iterable[Validator] = (), hooks: Mapping[str, Callable[..., Any]] | None = None,
                 knowledge_service: Any | None = None,
                 replace_fn: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any] | None = None,
                 failure_injection: str | Iterable[str] | None = None):
        self.workspace = Path(workspace or Path.cwd()).resolve()
        # Standalone service callers historically became eligible after three
        # turns; the application facade passes its configured (usually six
        # turn) policy explicitly.
        self.policy = policy or EvolutionPolicy(
            min_turns=3,
            allowed_roots=list(allowed_roots or ALLOWED_ROOT_NAMES),
        )
        self.allowed_roots = list(allowed_roots or self.policy.allowed_roots)
        self.data_dir = Path(data_dir or (self.workspace / "data/langgraph/evolution")).resolve()
        self.store = EvolutionStore(self.data_dir)
        self.validators = list(validators)
        self.hooks = dict(hooks or {})
        self.knowledge_service = knowledge_service
        self._install_knowledge_hooks()
        self.replace_fn = replace_fn or os.replace
        if isinstance(failure_injection, str):
            self.failure_injection = {failure_injection}
        else:
            self.failure_injection = set(failure_injection or ())
        self._applied_ids: set[str] = set()
        self._due_seen: set[str] = set()
        self._knowledge_rag_before_by_id: dict[str, Any] = {}

    @staticmethod
    def _affected_wiki_paths(manifest: Iterable[Mapping[str, Any]]) -> list[str]:
        return [
            str(item["path"]).replace("\\", "/")
            for item in manifest
            if str(item.get("path", "")).replace("\\", "/").startswith("knowledge/")
        ]

    @staticmethod
    def _rebuild_report() -> dict[str, Any]:
        return {
            "wiki": "not_run",
            "rag": "not_run",
            "graph": "not_run",
            "errors": [],
        }

    @staticmethod
    def _check_rebuild_result(result: Any, name: str) -> None:
        if result is None:
            return
        if result is False:
            raise RuntimeError(f"{name} rebuild failed")
        status = getattr(result, "status", None)
        errors = getattr(result, "errors", None)
        if isinstance(result, Mapping):
            if result.get("ok") is False:
                raise RuntimeError(f"{name} rebuild failed")
            status = result.get("status", status)
            errors = result.get("errors", errors)
        if status not in (None, "ok", "success") or errors:
            raise RuntimeError(f"{name} rebuild failed: {errors or status}")

    def _knowledge_rag_rebuild(self, _evolution_id: str, manifest: list[dict[str, Any]]) -> None:
        """Initialize the six seeds before rebuilding derived Wiki state."""
        service = self.knowledge_service
        if service is None:
            return
        initializer = getattr(service, "initialize_seed_wiki", None)
        if not callable(initializer):
            initializer = getattr(service, "initialize", None)
        if callable(initializer):
            # Initialization is needed for a fresh Wiki, but must not
            # overwrite an evolved seed page before rebuilding its index.
            store = getattr(service, "store", None)
            pages = getattr(service, "pages", ())
            has_seed_wiki = bool(store is not None and pages and all(
                store.exists(f"orbital-dynamics/{page.slug}.md") for page in pages
            ))
            if not has_seed_wiki:
                self._check_rebuild_result(initializer(), "wiki initialization")

    def _knowledge_graph_rebuild(self, _evolution_id: str, manifest: list[dict[str, Any]]) -> None:
        service = self.knowledge_service
        if service is None:
            return
        paths = self._affected_wiki_paths(manifest)
        rebuild = getattr(service, "rebuild_derived", None)
        if not callable(rebuild):
            rebuild = getattr(service, "rebuild", None)
        if callable(rebuild):
            try:
                result = rebuild(paths)
            except TypeError:
                result = rebuild()
            self._check_rebuild_result(result, "derived knowledge")
            return
        # Compatibility with injected services exposing separate boundaries.
        for name in ("rebuild_index", "index", "rebuild_graph", "graph"):
            callback = getattr(service, name, None)
            if callable(callback):
                try:
                    result = callback(paths)
                except TypeError:
                    result = callback()
                self._check_rebuild_result(result, name)

    def _install_knowledge_hooks(self) -> None:
        if self.knowledge_service is None:
            return
        self.hooks.setdefault("rag_rebuild", self._knowledge_rag_rebuild)
        self.hooks.setdefault("graph_rebuild", self._knowledge_graph_rebuild)

    def _snapshot_tree(self, root: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if not root.exists():
            return
        for source in root.rglob("*"):
            if not source.is_file():
                continue
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _restore_tree(self, root: Path, snapshot: Path) -> None:
        if not snapshot.exists():
            return
        root.mkdir(parents=True, exist_ok=True)
        for existing in list(root.rglob("*")):
            if existing.is_file():
                existing.unlink()
        for source in snapshot.rglob("*"):
            if source.is_file():
                target = root / source.relative_to(snapshot)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def _capture_knowledge_state(self, evolution_id: str) -> None:
        service = self.knowledge_service
        if service is None:
            return
        tx = self.store.transaction_dir(evolution_id)
        wiki_root = Path(getattr(service, "wiki_root", self.workspace / "knowledge")).resolve()
        data_root = Path(getattr(service, "data_dir", self.workspace / "data/langgraph/rag")).resolve()
        self._snapshot_tree(wiki_root, tx / "knowledge-before" )
        self._snapshot_tree(data_root, tx / "rag-before")
        self.store.write_json(evolution_id, "knowledge-state.json", {
            "wiki_root": str(wiki_root), "data_root": str(data_root),
        })
        self._knowledge_rag_before_by_id[evolution_id] = getattr(service, "rag", None)

    def _restore_knowledge_state(self, evolution_id: str) -> None:
        service = self.knowledge_service
        if service is None:
            return
        tx = self.store.transaction_dir(evolution_id)
        try:
            state = self.store.read_json(evolution_id, "knowledge-state.json")
        except (FileNotFoundError, ValueError):
            state = {}
        wiki_root = Path(state.get("wiki_root", getattr(service, "wiki_root", self.workspace / "knowledge")))
        data_root = Path(state.get("data_root", getattr(service, "data_dir", self.workspace / "data/langgraph/rag")))
        self._restore_tree(wiki_root, tx / "knowledge-before")
        self._restore_tree(data_root, tx / "rag-before")
        previous = self._knowledge_rag_before_by_id.get(evolution_id)
        try:
            from aerospace_agent.rag.aerospace_rag import AerospaceRAG
            # Reload persisted artifacts to discard any in-memory mutation a
            # failing hook may have made.  If the service had no persisted
            # index yet, preserve its pre-transaction object instead.
            if any((tx / "rag-before").rglob("*")):
                service.rag = AerospaceRAG(data_dir=str(data_root), autoload=True, auto_default_knowledge=False)
            elif previous is not None:
                service.rag = previous
        except Exception:
            if previous is not None:
                service.rag = previous

    def _record(self, evolution_id: str, proposal: EvolutionProposal, status: str = "proposed", **extra: Any) -> EvolutionRecord:
        tx = self.store.transaction_dir(evolution_id)
        history = [str(item.get("state")) for item in self.store.read_journal(evolution_id)]
        manifest = extra.pop("manifest", extra.get("before_manifest", []))
        record = EvolutionRecord(evolution_id=evolution_id, thread_id=proposal.thread_id, run_id=proposal.run_id,
                                 status=status, proposal=proposal, state_history=history, manifest=manifest,
                                 proposal_path=tx / "proposal.json", manifest_path=tx / "manifest.json",
                                 checkpoint_id=proposal.checkpoint_id, source=dict(proposal.source), **extra)
        # Keep a stable location for callers while retaining all validation
        # details in the existing field for backwards compatibility.
        if not record.rebuild and isinstance(extra.get("validation_details"), Mapping):
            record.rebuild = dict(extra["validation_details"].get("rebuild", {}))
        return record

    @staticmethod
    def _proposal_json(proposal: EvolutionProposal) -> dict[str, Any]:
        return proposal.model_dump(mode="json")

    def _transition(self, evolution_id: str, status: str, report: dict[str, Any], **details: Any) -> None:
        # Journal before any action that follows the transition.
        event = self.store.append_transition(evolution_id, status, **details)
        report.setdefault("transitions", []).append(event)
        report["status"] = status
        self.store.write_json(evolution_id, "report.json", report)

    def _manifest_for(self, proposal: EvolutionProposal, evolution_id: str) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for index, change in enumerate(proposal.changes):
            candidate = validate_target_path(change.path, self.workspace, self.allowed_roots)
            if candidate.exists() and not candidate.is_file():
                raise ValueError(f"evolution target is not a regular file: {change.path}")
            prior_exists = candidate.exists()
            prior_bytes = candidate.read_bytes() if prior_exists else None
            prior_mode = (candidate.stat().st_mode & 0o7777) if prior_exists else None
            after_bytes = None if change.operation == "delete" else str(change.content or "").encode("utf-8")
            item = {
                "index": index,
                "path": str(change.path).replace("\\", "/"),
                "operation": change.operation,
                "prior_exists": prior_exists,
                "prior_bytes": encode_bytes(prior_bytes),
                "prior_mode": prior_mode,
                "mode": prior_mode,
                "before_sha256": sha256_bytes(prior_bytes) if prior_bytes is not None else None,
                "after_sha256": sha256_bytes(after_bytes) if after_bytes is not None else None,
                # Keep decoded data internal to facilitate exact rollback.
                "after_bytes": after_bytes,
            }
            manifest.append(item)
            stage_path = self.store.transaction_dir(evolution_id) / "staging" / Path(item["path"])
            if after_bytes is not None:
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                stage_path.write_bytes(after_bytes)
                if prior_mode is not None:
                    os.chmod(stage_path, prior_mode)
        return manifest

    def _write_manifest(self, evolution_id: str, manifest: list[dict[str, Any]]) -> None:
        serializable = []
        for item in manifest:
            copy = dict(item)
            if isinstance(copy.get("after_bytes"), bytes):
                copy["after_bytes_b64"] = encode_bytes(copy.pop("after_bytes"))
            serializable.append(copy)
        self.store.write_json(evolution_id, "manifest.json", {"targets": serializable})

    def _backup(self, evolution_id: str, manifest: list[dict[str, Any]]) -> None:
        base = self.store.transaction_dir(evolution_id) / "backup"
        for item in manifest:
            prior = decode_bytes(item.get("prior_bytes"))
            if prior is None:
                continue
            destination = base / Path(item["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(prior)
            if item.get("prior_mode") is not None:
                os.chmod(destination, int(item["prior_mode"]))

    def _invoke_hook(self, name: str, evolution_id: str, manifest: list[dict[str, Any]], *, ignore_injection: bool = False) -> None:
        if not ignore_injection and name in self.failure_injection:
            raise RuntimeError(f"injected failure at {name}")
        callback = self.hooks.get(name)
        if callback is None:
            if name == "wiki_log":
                self._append_log(evolution_id)
            return
        # Select a compatible arity before invocation.  Retrying every
        # ``TypeError`` would misclassify a real callback bug as a signature
        # mismatch and could hide the failure that must trigger rollback.
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            signature = None
        candidates = ((evolution_id, manifest), (evolution_id,), ())
        if signature is not None:
            for args in candidates:
                try:
                    signature.bind(*args)
                except TypeError:
                    continue
                callback(*args)
                return
            raise TypeError(f"hook {name!r} has an unsupported signature")
        callback(evolution_id, manifest)

    def _append_log(self, evolution_id: str) -> None:
        path = self.workspace / "knowledge" / "log.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        line = f"- evolution {evolution_id}\n"
        if line not in text:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                if text and not text.endswith("\n"):
                    handle.write("\n")
                handle.write(line)

    def _remove_log(self, evolution_id: str) -> None:
        path = self.workspace / "knowledge" / "log.md"
        if not path.exists():
            return
        line = f"- evolution {evolution_id}\n"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(item for item in lines if item != line), encoding="utf-8")

    def _apply_target(self, item: dict[str, Any], *, fail_at: int | None, counter: list[int]) -> None:
        counter[0] += 1
        if fail_at is not None and counter[0] == int(fail_at):
            raise RuntimeError(f"injected failure at operation {counter[0]}")
        path = self.workspace / Path(item["path"])
        if item["operation"] == "delete":
            if path.exists():
                path.unlink()
            return
        content = item.get("after_bytes")
        assert isinstance(content, bytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".evolution", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if item.get("prior_mode") is not None:
                os.chmod(name, int(item["prior_mode"]))
            self.replace_fn(name, path)
            if item.get("prior_mode") is not None:
                os.chmod(path, int(item["prior_mode"]))
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def _restore(self, manifest: list[dict[str, Any]]) -> None:
        for item in reversed(manifest):
            path = self.workspace / Path(item["path"])
            prior = decode_bytes(item.get("prior_bytes"))
            if prior is None:
                if path.exists() and path.is_file():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".rollback", dir=str(path.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(prior)
                    handle.flush()
                    os.fsync(handle.fileno())
                if item.get("prior_mode") is not None:
                    os.chmod(name, int(item["prior_mode"]))
                os.replace(name, path)
                if item.get("prior_mode") is not None:
                    os.chmod(path, int(item["prior_mode"]))
            finally:
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass

    def apply(self, proposal: EvolutionProposal | Mapping[str, Any], *, fail_at: int | None = None,
              failure_injection: str | Iterable[str] | None = None,
              replace_fn: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any] | None = None) -> EvolutionRecord:
        if not isinstance(proposal, EvolutionProposal):
            try:
                proposal = EvolutionProposal.model_validate(proposal)
            except Exception as exc:
                raise ValueError("invalid evolution proposal") from exc
        # Path policy failures are caller errors, not transaction failures.
        for change in proposal.changes:
            normalized = str(change.path).replace("\\", "/").strip("/")
            if any(
                normalized == root or normalized.startswith(f"{root}/")
                for root in self.CORE_MANAGED_ROOTS
            ):
                raise PermissionError(
                    "skill and workflow activation must use the Agent Core signed trust chain"
                )
            validate_target_path(change.path, self.workspace, self.allowed_roots)
        evolution_id = uuid.uuid4().hex
        self.store.transaction_dir(evolution_id)
        report: dict[str, Any] = {
            "evolution_id": evolution_id,
            "status": "proposed",
            "errors": [],
            "transitions": [],
            "rebuild": self._rebuild_report(),
        }
        report["rebuild_status"] = report["rebuild"]
        self.store.write_json(evolution_id, "proposal.json", self._proposal_json(proposal))
        self._transition(evolution_id, "proposed", report)
        manifest: list[dict[str, Any]] = []
        old_failure = self.failure_injection
        old_replace = self.replace_fn
        if failure_injection is not None:
            self.failure_injection = {failure_injection} if isinstance(failure_injection, str) else set(failure_injection)
        if replace_fn is not None:
            self.replace_fn = replace_fn
        try:
            # Journal the intent before touching staging files.
            self._transition(evolution_id, "staged", report)
            manifest = self._manifest_for(proposal, evolution_id)
            self._write_manifest(evolution_id, manifest)
            if self.knowledge_service is not None and self._affected_wiki_paths(manifest):
                self._capture_knowledge_state(evolution_id)
            # Journal the backup phase before copying any prior bytes.
            self._transition(evolution_id, "backed_up", report)
            self._backup(evolution_id, manifest)
            self._transition(evolution_id, "validating", report)
            results = run_validators(proposal, self.workspace, manifest, self.validators)
            report["validation_results"] = [result.as_dict() for result in results]
            self.store.write_json(evolution_id, "report.json", report)
            if any(not result.ok for result in results):
                report["errors"].extend(result.message for result in results if not result.ok)
                self._transition(evolution_id, "validation_failed", report)
                self._restore(manifest)
                self._transition(evolution_id, "rolled_back", report)
                return self._record(evolution_id, proposal, "rolled_back", before_manifest=manifest,
                                    after_manifest=[], validation_results=report["validation_results"],
                                    validation_details={"rebuild": report["rebuild"]}, rebuild=report["rebuild"],
                                    report_path=self.store.transaction_dir(evolution_id) / "report.json")
            counter = [0]
            for item in manifest:
                self._apply_target(item, fail_at=fail_at, counter=counter)
            affected_wiki = bool(self._affected_wiki_paths(manifest))
            for hook in ("wiki_log", "affected_test", "rag_rebuild", "graph_rebuild"):
                if hook in {"rag_rebuild", "graph_rebuild"} and not affected_wiki:
                    continue
                if hook in {"rag_rebuild", "graph_rebuild"}:
                    report["rebuild"][hook.replace("_rebuild", "")] = "running"
                    self.store.write_json(evolution_id, "report.json", report)
                self._invoke_hook(hook, evolution_id, manifest)
                if hook in {"rag_rebuild", "graph_rebuild"}:
                    report["rebuild"][hook.replace("_rebuild", "")] = "ok"
                    self.store.write_json(evolution_id, "report.json", report)
                elif hook == "wiki_log" and affected_wiki:
                    report["rebuild"]["wiki"] = "ok"
                    self.store.write_json(evolution_id, "report.json", report)
            report["after_manifest"] = [{"path": item["path"], "sha256": item["after_sha256"]} for item in manifest]
            self._transition(evolution_id, "committed", report)
            self._applied_ids.add(evolution_id)
            return self._record(evolution_id, proposal, "committed", before_manifest=manifest,
                                after_manifest=report["after_manifest"], validation_results=report.get("validation_results", []),
                                validation_details={"rebuild": report["rebuild"]}, rebuild=report["rebuild"],
                                report_path=self.store.transaction_dir(evolution_id) / "report.json")
        except Exception as exc:
            report["errors"].append(str(exc))
            if report.get("rebuild"):
                report["rebuild"]["errors"].append(str(exc))
                for key in ("rag", "graph"):
                    if report["rebuild"].get(key) == "running":
                        report["rebuild"][key] = "failed"
            try:
                self._transition(evolution_id, "commit_failed", report, error=str(exc))
                self._restore(manifest)
                if self.knowledge_service is not None and self._affected_wiki_paths(manifest):
                    self._restore_knowledge_state(evolution_id)
                self._remove_log(evolution_id)
                self._transition(evolution_id, "rolled_back", report)
            except Exception as rollback_exc:
                report["errors"].append(f"rollback failed: {rollback_exc}")
                self.store.write_json(evolution_id, "report.json", report)
            return self._record(evolution_id, proposal, "rolled_back", before_manifest=manifest,
                                validation_results=report.get("validation_results", []),
                                validation_details={"rebuild": report["rebuild"]}, rebuild=report["rebuild"],
                                report_path=self.store.transaction_dir(evolution_id) / "report.json")
        finally:
            self.failure_injection = old_failure
            self.replace_fn = old_replace

    def rollback(self, evolution_id: str) -> EvolutionRecord:
        report = self.store.read_json(evolution_id, "report.json")
        report.setdefault("rebuild", self._rebuild_report())
        report["rebuild_status"] = report["rebuild"]
        proposal = EvolutionProposal.model_validate(self.store.read_json(evolution_id, "proposal.json"))
        manifest_serialized = self.store.read_json(evolution_id, "manifest.json").get("targets", [])
        manifest: list[dict[str, Any]] = []
        for item in manifest_serialized:
            copy = dict(item)
            copy["prior_bytes"] = copy.get("prior_bytes")
            copy["after_bytes"] = decode_bytes(copy.pop("after_bytes_b64", None))
            manifest.append(copy)
        self._transition(evolution_id, "rollback_requested", report)
        for item in manifest:
            path = self.workspace / Path(item["path"])
            current = path.read_bytes() if path.exists() and path.is_file() else None
            current_hash = sha256_bytes(current) if current is not None else None
            if current_hash != item.get("after_sha256"):
                report["errors"].append(f"hash conflict: {item['path']}")
                self._transition(evolution_id, "conflict", report)
                return self._record(evolution_id, proposal, "conflict", before_manifest=manifest,
                                    validation_details={"rebuild": report["rebuild"]}, rebuild=report["rebuild"],
                                    report_path=self.store.transaction_dir(evolution_id) / "report.json")
        self._restore(manifest)
        try:
            if self.knowledge_service is not None and self._affected_wiki_paths(manifest):
                self._restore_knowledge_state(evolution_id)
                report["rebuild"]["rag"] = "restored"
                report["rebuild"]["graph"] = "restored"
            else:
                self._invoke_hook("rag_rebuild", evolution_id, manifest, ignore_injection=True)
                report["rebuild"]["rag"] = "ok"
        except Exception as exc:
            report["rebuild"]["errors"].append(str(exc))
            report["errors"].append(str(exc))
            report["rebuild"]["rag"] = "failed"
            self.store.write_json(evolution_id, "report.json", report)
            raise
        self._append_log(evolution_id)
        self._transition(evolution_id, "rolled_back", report)
        return self._record(evolution_id, proposal, "rolled_back", before_manifest=manifest,
                            validation_details={"rebuild": report["rebuild"]}, rebuild=report["rebuild"],
                            report_path=self.store.transaction_dir(evolution_id) / "report.json")

    manual_rollback = rollback
    request_rollback = rollback

    def is_due(self, **kwargs: Any) -> bool:
        return self.policy.is_due(**kwargs).due

    def evolve_due(self, proposal: EvolutionProposal | Mapping[str, Any] | None, *, due_key: str | None = None,
                   **eligibility: Any) -> EvolutionRecord | None:
        key = due_key or (proposal.run_id if isinstance(proposal, EvolutionProposal) else str((proposal or {}).get("run_id", "")))
        if key in self._due_seen or self._durable_due_seen(key):
            return None
        if not self.policy.is_due(already_applied=False, **eligibility).due:
            return None
        self._due_seen.add(key)
        if proposal is None:
            return None
        parsed = proposal if isinstance(proposal, EvolutionProposal) else parse_llm_proposal(proposal)
        if parsed is None:
            return None
        if parsed.unfinished_items:
            pending = self.workspace / "memory" / "pending.md"
            pending.parent.mkdir(parents=True, exist_ok=True)
            with pending.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(f"- {item}" for item in parsed.unfinished_items) + "\n")
            return None
        return self.apply(parsed)

    def _durable_due_seen(self, key: str) -> bool:
        if not key:
            return False
        try:
            for tx in self.data_dir.iterdir():
                if not tx.is_dir():
                    continue
                proposal_path = tx / "proposal.json"
                report_path = tx / "report.json"
                if not proposal_path.exists() or not report_path.exists():
                    continue
                import json
                if str(json.loads(proposal_path.read_text(encoding="utf-8")).get("run_id", "")) != key:
                    continue
                if str(json.loads(report_path.read_text(encoding="utf-8")).get("status", "")) in {"committed", "rolled_back", "conflict"}:
                    return True
        except OSError:
            return False
        return False


__all__ = ["EvolutionService", "EvolutionPolicy", "ValidationResult", "parse_llm_proposal"]
