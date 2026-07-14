"""Manifest-only capability discovery.

The registry deliberately owns no Python handlers. It is safe to pass its
results to routing and planning because every returned value is a defensive
copy of a validated :class:`CapabilityManifest`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import yaml
from pydantic import Field

from .models import CapabilityManifest, ContractModel


PLANNER_CANDIDATE_LIMIT = 12
ALLOWED_IMPORT_ROOTS = (
    "aerospace_agent.mcp.tools",
    "aerospace_agent.integrations",
    "aerospace_agent.domains",
)


class PlannerCandidate(ContractModel):
    """Bounded planner view; executor names are the units counted by the limit."""

    capability_id: str
    executor_names: list[str] = Field(min_length=1, max_length=12)


class CapabilityRegistry:
    """Validate, discover and filter capability manifests without executing."""

    def __init__(
        self,
        manifests: Iterable[CapabilityManifest],
        *,
        approval_verifier: Callable[[CapabilityManifest], bool] | None = None,
    ) -> None:
        self._manifests: dict[str, CapabilityManifest] = {}
        self._approval_verifier = approval_verifier
        executor_owners: dict[str, str] = {}
        for item in manifests:
            manifest = CapabilityManifest.model_validate(item.model_dump(mode="python"))
            if not any(
                manifest.source == root or manifest.source.startswith(f"{root}.")
                for root in ALLOWED_IMPORT_ROOTS
            ):
                raise ValueError(
                    f"capability source must be under allowed current-repository roots: {manifest.source}"
                )
            requires_approval = self._requires_live_approval(manifest)
            if manifest.status == "available" and requires_approval:
                if approval_verifier is None or not approval_verifier(self._copy(manifest)):
                    raise ValueError(
                        f"available workflow/integration requires trusted approval verification: "
                        f"{manifest.capability_id}"
                    )
            if manifest.capability_id in self._manifests:
                raise ValueError(f"duplicate capability_id: {manifest.capability_id}")
            for executor_name in manifest.tool_names:
                owner = executor_owners.get(executor_name)
                if owner is not None:
                    raise ValueError(
                        f"duplicate executor name {executor_name}: owned by {owner} and "
                        f"{manifest.capability_id}"
                    )
                executor_owners[executor_name] = manifest.capability_id
            self._manifests[manifest.capability_id] = manifest

    @classmethod
    def from_repository(
        cls,
        workspace_root: str | Path,
        manifest_paths: Sequence[str | Path],
        *,
        approval_verifier: Callable[[CapabilityManifest], bool] | None = None,
    ) -> "CapabilityRegistry":
        """Load YAML manifests only from explicitly supplied workspace paths."""

        root = Path(workspace_root).resolve()
        manifests: list[CapabilityManifest] = []
        for raw_path in manifest_paths:
            path = Path(raw_path).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"manifest path is outside workspace: {path}") from exc
            if path.suffix.casefold() not in {".yaml", ".yml"}:
                raise ValueError(f"capability manifest must be YAML: {path}")
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            documents = loaded if isinstance(loaded, list) else [loaded]
            for document in documents:
                manifests.append(CapabilityManifest.model_validate(document))
        return cls(manifests, approval_verifier=approval_verifier)

    @staticmethod
    def _copy(manifest: CapabilityManifest) -> CapabilityManifest:
        return CapabilityManifest.model_validate(manifest.model_dump(mode="python"))

    @staticmethod
    def _requires_live_approval(manifest: CapabilityManifest) -> bool:
        return manifest.category == "workflow" or (
            manifest.source == "aerospace_agent.integrations"
            or manifest.source.startswith("aerospace_agent.integrations.")
        )

    def _current(self, manifest: CapabilityManifest) -> CapabilityManifest:
        """Fail a cached route closed when its approval or artifacts drift."""

        current = self._copy(manifest)
        if current.status != "available" or not self._requires_live_approval(current):
            return current
        try:
            approved = self._approval_verifier is not None and self._approval_verifier(
                self._copy(current)
            )
        except Exception:
            approved = False
        if not approved:
            current.status = "unavailable"
        return current

    def list_manifests(self) -> list[CapabilityManifest]:
        """Return all discovery descriptors, including ``interface_only`` ones."""

        return [self._current(item) for item in self._manifests.values()]

    def get(self, capability_id: str) -> CapabilityManifest:
        try:
            manifest = self._manifests[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability_id: {capability_id}") from exc
        return self._current(manifest)

    def discover(self, capability_id: str) -> CapabilityManifest:
        """Return a descriptor by ID; discovery does not imply executability."""

        return self.get(capability_id)

    def find_by_tool_name(self, tool_name: str) -> CapabilityManifest | None:
        for manifest in self._manifests.values():
            if tool_name in manifest.tool_names:
                return self._current(manifest)
        return None

    def candidates_for_intents(
        self,
        intents: Sequence[str],
        *,
        limit: int = PLANNER_CANDIDATE_LIMIT,
    ) -> list[CapabilityManifest]:
        """Return relevant executable manifests, capped at the planner limit."""

        if not 1 <= limit <= PLANNER_CANDIDATE_LIMIT:
            raise ValueError("candidate limit must be between 1 and 12")
        requested = {item.strip().casefold() for item in intents if item.strip()}
        if not requested:
            return []

        ranked: list[tuple[int, int, CapabilityManifest]] = []
        for order, manifest in enumerate(self._manifests.values()):
            manifest = self._current(manifest)
            if manifest.status != "available":
                continue
            provided = {item.strip().casefold() for item in manifest.intents if item.strip()}
            score = len(requested & provided)
            if score:
                ranked.append((-score, order, manifest))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [self._copy(item[2]) for item in ranked[:limit]]

    def available_manifests(self, *, limit: int = PLANNER_CANDIDATE_LIMIT) -> list[CapabilityManifest]:
        """Return a bounded discovery view for an ambiguity classifier."""

        if not 1 <= limit <= PLANNER_CANDIDATE_LIMIT:
            raise ValueError("candidate limit must be between 1 and 12")
        return [
            self._current(item)
            for item in self._manifests.values()
            if self._current(item).status == "available"
        ][:limit]

    @staticmethod
    def _executor_names(manifest: CapabilityManifest) -> list[str]:
        return list(manifest.tool_names) or [manifest.capability_id]

    def planner_candidates_for_intents(
        self,
        intents: Sequence[str],
        *,
        limit: int = PLANNER_CANDIDATE_LIMIT,
    ) -> list[PlannerCandidate]:
        """Return candidates while counting tools/workflows, not manifest groups."""

        if not 1 <= limit <= PLANNER_CANDIDATE_LIMIT:
            raise ValueError("candidate limit must be between 1 and 12")
        manifests = self.candidates_for_intents(intents, limit=PLANNER_CANDIDATE_LIMIT)
        return self._budget_candidates(manifests, limit=limit)

    def planner_candidates_for_request(
        self,
        message: str,
        *,
        limit: int = PLANNER_CANDIDATE_LIMIT,
    ) -> list[PlannerCandidate]:
        """Select available candidates whose declared intent or name occurs in the request."""

        if not 1 <= limit <= PLANNER_CANDIDATE_LIMIT:
            raise ValueError("candidate limit must be between 1 and 12")
        normalized = message.casefold()
        scored: list[tuple[int, int, CapabilityManifest, list[str]]] = []
        for order, manifest in enumerate(self._manifests.values()):
            manifest = self._current(manifest)
            if manifest.status != "available":
                continue
            matched_intents = [
                intent
                for intent in manifest.intents
                if intent.strip() and intent.casefold() in normalized
            ]
            matched_executors = [
                name
                for name in self._executor_names(manifest)
                if name.strip() and name.casefold() in normalized
            ]
            score = len(matched_intents) + len(matched_executors)
            if score:
                executor_names = (
                    matched_executors if matched_executors else self._executor_names(manifest)
                )
                scored.append((-score, order, manifest, executor_names))
        scored.sort(key=lambda item: (item[0], item[1]))
        remaining = limit
        candidates: list[PlannerCandidate] = []
        for _, _, manifest, executor_names in scored:
            if remaining == 0:
                break
            selected_names = executor_names[:remaining]
            candidates.append(
                PlannerCandidate(
                    capability_id=manifest.capability_id,
                    executor_names=selected_names,
                )
            )
            remaining -= len(selected_names)
        return candidates

    def manifests_for_planner_candidates(
        self,
        candidates: Sequence[PlannerCandidate],
    ) -> list[CapabilityManifest]:
        """Create defensive manifest views containing only budgeted executor names."""

        views: list[CapabilityManifest] = []
        for candidate in candidates:
            manifest = self.get(candidate.capability_id)
            if manifest.tool_names:
                manifest.tool_names = list(candidate.executor_names)
            views.append(manifest)
        return views

    def _budget_candidates(
        self,
        manifests: Sequence[CapabilityManifest],
        *,
        limit: int,
    ) -> list[PlannerCandidate]:
        remaining = limit
        candidates: list[PlannerCandidate] = []
        for manifest in manifests:
            if remaining == 0:
                break
            names = self._executor_names(manifest)[:remaining]
            if names:
                candidates.append(
                    PlannerCandidate(
                        capability_id=manifest.capability_id,
                        executor_names=names,
                    )
                )
                remaining -= len(names)
        return candidates


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "CapabilityRegistry",
    "PLANNER_CANDIDATE_LIMIT",
    "PlannerCandidate",
]
