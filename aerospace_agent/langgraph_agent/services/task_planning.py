"""LLM-backed TaskPlan construction and deterministic completion assessment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..agent_core.models import CheckResult, TaskPlan
from ..agent_core.planning import build_task_plan
from ..agent_core.review import ReviewAssessment
from ..prompts import AEROSPACE_ASSISTANT_IDENTITY


def _message_text(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("messages", []) or []):
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return str(state.get("user_message", "") or "")


def _json_object(text: Any) -> Mapping[str, Any] | None:
    match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _canonical_hash(value: Any) -> str:
    dumped = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class LLMTaskPlanService:
    """Ask the active model for a bounded plan, then validate every binding."""

    _READ_ONLY_TOOL_NAMES = frozenset(
        {
            "file.read",
            "file.read_lines",
            "file.list",
            "file.search",
            "file.info",
            "terminal.status",
            "browser.open",
            "browser.follow_link",
            "browser.extract",
            "web.search",
            "web.fetch",
            "schedule.list",
            "memory.search",
            "memory.list",
            "git.status",
            "git.diff",
            "git.log",
            "git.branch_info",
            "workflow.list",
            "capability.list",
            "capability.describe",
        }
    )

    def __init__(self, llm: Any, capability_registry: Any, execution_registry: Any) -> None:
        self.llm = llm
        self.capability_registry = capability_registry
        self.execution_registry = execution_registry

    def create_task_plan(self, *, route: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
        del route
        if self.llm is None or not callable(getattr(self.llm, "chat", None)):
            return None
        request = _message_text(state)
        manifests = [
            manifest
            for manifest in self.capability_registry.list_manifests()
            if manifest.status == "available"
            and manifest.category in {"basic", "space_basic"}
            and manifest.tool_names
        ]
        tools = [name for manifest in manifests for name in manifest.tool_names]
        prompt = (
            "Decompose this aerospace request into a short executable TaskPlan. "
            "Return only one JSON object: "
            '{"steps":[{"step_id":"step-1","tool_name":"file.read",'
            '"inputs":{},"dependencies":[]}]}. '
            "Use only the available tool names. Each step must be read-only or "
            "explicitly require confirmation; never invent a domain tool. "
            "Keep the plan to at most 6 steps and preserve the user's requested order.\n\n"
            f"Request: {request}\n"
            f"Available tools: {json.dumps(tools, ensure_ascii=False)}"
        )
        try:
            response = self.llm.chat(
                prompt,
                system_prompt=(
                    f"{AEROSPACE_ASSISTANT_IDENTITY}\n"
                    "你现在只负责生成严格 JSON 的执行计划，不要回答用户，不要输出 Markdown。"
                ),
                max_tokens=900,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
            response_text = str(response or "")
            # Test/dry-run echo clients return the prompt verbatim.  Treating
            # the schema example in that prompt as a model plan would execute
            # an unbound placeholder, so only parse an actual model response.
            payload = (
                None
                if response_text.lstrip().startswith("echo:") or prompt in response_text
                else _json_object(response_text)
            )
            raw_steps = payload.get("steps") if payload is not None else None
            steps = self._normalize_steps(raw_steps, manifests)
        except Exception:
            steps = []
        if not steps:
            steps = self._fallback_steps(request, manifests)
        if not steps:
            # The caller turns this into a structured partial result.  A model
            # that did not produce a valid, bound plan must never execute.
            return None
        return {
            "task_plan": self._build_plan(state, request, steps),
            "retrieval_request": None,
        }

    def _normalize_steps(self, raw_steps: Any, manifests: Sequence[Any]) -> list[dict[str, Any]]:
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 6:
            return []
        by_tool = {
            tool: manifest
            for manifest in manifests
            for tool in manifest.tool_names
        }
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_steps, 1):
            if not isinstance(raw, Mapping):
                return []
            tool_name = str(raw.get("tool_name", "")).strip()
            manifest = by_tool.get(tool_name)
            if manifest is None:
                return []
            step_id = str(raw.get("step_id") or f"step-{index}").strip()
            if not step_id or step_id in seen:
                return []
            seen.add(step_id)
            dependencies = raw.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                str(item) not in seen for item in dependencies
            ):
                return []
            inputs = raw.get("inputs", raw.get("arguments", {}))
            if not isinstance(inputs, Mapping):
                return []
            result.append(
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "capability": manifest.capability_id,
                    "inputs": dict(inputs),
                    "dependencies": [str(item) for item in dependencies],
                    "risk_level": manifest.risk_level,
                    "requires_confirmation": (
                        manifest.risk_level != "read_only"
                        or tool_name not in self._READ_ONLY_TOOL_NAMES
                    ),
                }
            )
        return result

    def _fallback_steps(self, request: str, manifests: Sequence[Any]) -> list[dict[str, Any]]:
        """Keep common local workflows usable after a malformed model response."""

        available = {tool: manifest for manifest in manifests for tool in manifest.tool_names}
        lower = request.casefold()
        steps: list[dict[str, Any]] = []
        if "agents.md" in lower and "file.read" in available:
            manifest = available["file.read"]
            steps.append(
                {
                    "step_id": "step-1",
                    "tool_name": "file.read",
                    "capability": manifest.capability_id,
                    "inputs": {"path": "AGENTS.md"},
                    "dependencies": [],
                    "risk_level": manifest.risk_level,
                    "requires_confirmation": "file.read" not in self._READ_ONLY_TOOL_NAMES,
                }
            )
        if ("python" in lower or "terminal" in lower or "command" in lower) and "terminal.run" in available:
            manifest = available["terminal.run"]
            steps.append(
                {
                    "step_id": f"step-{len(steps) + 1}",
                    "tool_name": "terminal.run",
                    "capability": manifest.capability_id,
                    "inputs": {"argv": ["python", "--version"]},
                    "dependencies": [steps[-1]["step_id"]] if steps else [],
                    "risk_level": manifest.risk_level,
                    "requires_confirmation": "terminal.run" not in self._READ_ONLY_TOOL_NAMES,
                }
            )
        return steps

    def _build_plan(
        self,
        state: Mapping[str, Any],
        request: str,
        steps: Sequence[Mapping[str, Any]],
    ) -> TaskPlan:
        snapshots = {
            str(step["capability"]): self.execution_registry.snapshot(str(step["capability"]))
            for step in steps
        }
        snapshot_payload = [item.model_dump(mode="json") for item in snapshots.values()]
        plan_steps = []
        for step in steps:
            step_id = str(step["step_id"])
            canonical_inputs = self._canonical_inputs(
                capability=str(step["capability"]),
                tool_name=str(step["tool_name"]),
                inputs=dict(step.get("inputs", {})),
            )
            plan_steps.append(
                {
                    "step_id": step_id,
                    "title": f"Execute {step['tool_name']}",
                    "description": f"Run {step['tool_name']} within the approved workspace boundary.",
                    "dependencies": list(step.get("dependencies", [])),
                    "executor_type": "space_basic_tool"
                    if snapshots[str(step["capability"])].capability_id.startswith("space.")
                    else "basic_tool",
                    "capability": str(step["capability"]),
                    "tool_name": str(step["tool_name"]),
                    "inputs": canonical_inputs,
                    "expected_outputs": ["successful tool result"],
                    "verification": [
                        {
                            "check_id": f"verify:{step_id}",
                            "description": "tool returned a successful result",
                            "method": "tool",
                            "required": True,
                            "acceptance_rule": "status == success",
                        }
                    ],
                    "risk_level": str(step["risk_level"]),
                    "requires_confirmation": bool(step["requires_confirmation"]),
                    "checkpoint_required": True,
                    "max_attempts": 2,
                }
            )
        project_id = str(state.get("project_id", ""))
        thread_id = str(state.get("thread_id", ""))
        root_run_id = str(state.get("root_run_id") or state.get("run_id") or "")
        now = datetime.now(UTC).isoformat()
        return build_task_plan(
            {
                "plan_id": f"plan:{root_run_id}:{uuid4().hex[:8]}",
                "project_id": project_id,
                "thread_id": thread_id,
                "root_run_id": root_run_id,
                "goal": {
                    "objective": request,
                    "success_criteria": ["all planned tool checks pass"],
                },
                "selected_capabilities": list(snapshots),
                "steps": plan_steps,
                "execution_snapshot": {
                    "capability_snapshots": snapshot_payload,
                    "registry_snapshot_sha256": _canonical_hash(snapshot_payload),
                    "captured_at": now,
                },
                "created_at": now,
            }
        )

    def _canonical_inputs(
        self,
        *,
        capability: str,
        tool_name: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Use the same defaults/path normalization as ExecutionRegistry."""

        registrations = getattr(self.execution_registry, "_registrations", {})
        registration = registrations.get(("tool", capability, tool_name))
        validator = getattr(self.execution_registry, "_validate_input_and_paths", None)
        if registration is None or not callable(validator):
            return dict(inputs)
        return dict(validator(registration, dict(inputs)))


class DeterministicReviewAssessor:
    """Turn durable tool results into the explicit checks ReviewService needs."""

    def assess(self, *, plan: TaskPlan, outcome: Any, evidence: Sequence[Any]) -> ReviewAssessment:
        del evidence
        results = dict(getattr(outcome, "step_results", {}) or {})
        checks: list[CheckResult] = []
        for step in plan.steps:
            result = results.get(step.step_id)
            success = result is not None and result.status == "success"
            audit_id = getattr(result, "audit_id", "") if result is not None else ""
            for verification in step.verification:
                checks.append(
                    CheckResult(
                        check_id=verification.check_id,
                        passed=success,
                        severity="info" if success else "critical",
                        message=("tool result satisfied the check" if success else "tool result did not satisfy the check"),
                        evidence_refs=[audit_id] if audit_id else [],
                    )
                )
        all_success = len(results) == len(plan.steps) and all(
            result.status == "success" for result in results.values()
        )
        return ReviewAssessment(
            goal_satisfied=all_success,
            boundary_compliant=True,
            constraints_satisfied=all_success,
            evidence_sufficient=all_success,
            tool_execution_safe=all_success,
            checks=checks,
            confidence=1.0 if all_success else 0.0,
        )


__all__ = ["DeterministicReviewAssessor", "LLMTaskPlanService"]
