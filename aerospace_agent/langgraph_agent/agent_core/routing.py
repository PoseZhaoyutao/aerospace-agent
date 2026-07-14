"""Rule-first capability routing with a strictly validated model fallback."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from .capabilities import CapabilityRegistry, PLANNER_CANDIDATE_LIMIT, PlannerCandidate
from .models import CapabilityGap, CapabilityManifest, ContractModel


RouteName = Literal[
    "conversation",
    "knowledge_qa",
    "direct_execution",
    "complex_task",
    "memory_operation",
    "project_operation",
    "clarify",
]


class CapabilityRoute(ContractModel):
    """Auditable output contract for every routing decision."""

    route: RouteName
    intent: str = Field(min_length=1)
    candidate_capability_ids: list[str] = Field(default_factory=list, max_length=12)
    candidate_executor_names: list[str] = Field(default_factory=list, max_length=12)
    selected_capability_id: str | None = None
    selected_executor_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    capability_gap: CapabilityGap | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if len(set(self.candidate_capability_ids)) != len(self.candidate_capability_ids):
            raise ValueError("candidate capability IDs must be unique")
        if len(set(self.candidate_executor_names)) != len(self.candidate_executor_names):
            raise ValueError("candidate executor names must be unique")
        if self.selected_capability_id is not None:
            if self.selected_capability_id not in self.candidate_capability_ids:
                raise ValueError("selected capability must be present in candidates")
        if self.selected_executor_name is not None:
            if self.selected_executor_name not in self.candidate_executor_names:
                raise ValueError("selected executor must be present in candidates")
            if self.selected_capability_id is None:
                raise ValueError("selected executor requires a selected capability")
        if self.route == "direct_execution":
            if self.selected_capability_id is None or self.selected_executor_name is None:
                raise ValueError("direct_execution requires selected capability and executor")
        if self.route in {"conversation", "knowledge_qa"}:
            if self.selected_capability_id is not None or self.selected_executor_name is not None:
                raise ValueError(f"{self.route} must not select an executor")
        if self.capability_gap is not None:
            if self.route != "clarify":
                raise ValueError("capability gap routes must clarify")
            if (
                self.candidate_capability_ids
                or self.candidate_executor_names
                or self.selected_capability_id is not None
                or self.selected_executor_name is not None
            ):
                raise ValueError("capability gap must not create execution candidates")
        return self


Classifier = Callable[[str, list[CapabilityManifest]], dict[str, object]]


class CapabilityRouter:
    """Apply deterministic operation rules before an optional model classifier."""

    _MEMORY_MARKERS = ("记忆", "memory")
    _MEMORY_ACTIONS = ("记住", "保存", "查询", "搜索", "修正", "更新", "遗忘", "忘记", "清空", "remember", "search", "update", "forget", "clear")
    _GIT_ACTIONS = ("git status", "git diff", "git log", "git branch", "git revert", "git restore", "创建检查点", "回滚")
    _SCHEDULE_MARKERS = ("定时", "schedule", "scheduler")
    _SCHEDULE_ACTIONS = ("创建", "新建", "列出", "查看", "取消", "运行", "create", "list", "cancel", "run")
    _COMPLEX_MARKERS = (
        "复杂任务", "任务拆解", "分解", "规划步骤", "工作规划", "工作流", "规划",
        "跨领域", "多步骤", "仿真", "模拟", "轨道传播", "传播工作流",
        "task plan", "workflow plan", "workflow", "simulation", "simulate",
        "orbit propagation", "propagation workflow", "two-body propagation",
    )
    _QUESTION_MARKERS = ("为什么", "是什么", "如何", "怎么", "解释")
    _CONVERSATION_MARKERS = ("你好", "您好", "hello", "hi", "谢谢", "再见")
    _FILE_RULES = (
        (("读取", "查看文件", "read file", "show file"), "file.read"),
        (("写入", "write file"), "file.write"),
        (("追加", "append"), "file.append"),
        (("列出文件", "list files"), "file.list"),
        (("搜索文件", "search files"), "file.search"),
        (("删除文件", "delete file"), "file.delete"),
        (("复制文件", "copy file"), "file.copy"),
        (("移动文件", "move file"), "file.move"),
        (("创建目录", "mkdir"), "file.mkdir"),
    )
    _NATURAL_FILE_VERBS = (
        ("read file", "file.read"),
        ("show file", "file.read"),
        ("list files", "file.list"),
        ("search files", "file.search"),
        ("write file", "file.write"),
        ("append file", "file.append"),
        ("delete file", "file.delete"),
        ("copy file", "file.copy"),
        ("move file", "file.move"),
        ("mkdir", "file.mkdir"),
        ("读取文件", "file.read"),
        ("查看文件", "file.read"),
        ("列出文件", "file.list"),
        ("搜索文件", "file.search"),
        ("写入文件", "file.write"),
        ("追加文件", "file.append"),
        ("删除文件", "file.delete"),
        ("复制文件", "file.copy"),
        ("移动文件", "file.move"),
        ("创建目录", "file.mkdir"),
    )
    _NATURAL_COMMAND_PREFIXES = (
        "terminal.run",
        "run command",
        "execute command",
        "run",
        "execute",
        "执行命令",
        "运行命令",
    )
    _KNOWN_COMMANDS = frozenset({"python", "python3", "python.exe", "git", "git.exe"})

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        direct_execution_confidence_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= direct_execution_confidence_threshold <= 1.0:
            raise ValueError("direct execution confidence threshold must be between 0 and 1")
        self._registry = registry
        self._direct_execution_confidence_threshold = direct_execution_confidence_threshold

    def prepare_request(
        self,
        message: str,
        *,
        requested_tool_name: str | None = None,
    ) -> tuple[str | None, dict[str, Any], bool]:
        """Extract only explicit, local request arguments for direct execution.

        This parser is intentionally conservative.  It never interprets free
        prose as a tool call, never accepts shell operators as structure, and
        leaves unparseable requests to the normal clarification/model path.
        Workspace containment and confirmation remain enforced by the concrete
        service and ``ExecutionRegistry``.
        """

        text = str(message or "").strip()
        if not text:
            return None, {}, False
        lowered = text.casefold()
        tool_name = (requested_tool_name or "").strip() or None
        remainder = text
        if tool_name and lowered.startswith(tool_name.casefold()):
            remainder = text[len(tool_name) :].strip(" \t:,-")
        elif tool_name is None:
            mentioned = self._mentioned_tools(lowered)
            if len(mentioned) == 1:
                tool_name = mentioned[0]
                remainder = text
                match = re.search(re.escape(tool_name), text, re.IGNORECASE)
                if match is not None:
                    remainder = text[match.end() :].strip(" \t:,-")

        if tool_name in {
            "file.read", "file.read_lines", "file.list", "file.search",
            "file.write", "file.append", "file.delete", "file.copy",
            "file.move", "file.mkdir",
        }:
            return self._prepare_file_request(tool_name, remainder)

        if tool_name == "terminal.run":
            return self._prepare_terminal_request(tool_name, remainder, explicit=True)

        if tool_name is None:
            for verb, candidate in self._NATURAL_FILE_VERBS:
                prefix = verb.casefold()
                if lowered == prefix or lowered.startswith(prefix + " "):
                    return self._prepare_file_request(
                        candidate,
                        text[len(verb) :].strip(" \t:,-"),
                    )
            for prefix in self._NATURAL_COMMAND_PREFIXES:
                normalized_prefix = prefix.casefold()
                if lowered == normalized_prefix or lowered.startswith(normalized_prefix + " "):
                    return self._prepare_terminal_request(
                        "terminal.run",
                        text[len(prefix) :].strip(),
                        explicit=normalized_prefix == "terminal.run",
                    )
            first = text.split(maxsplit=1)[0].casefold()
            if first in {"git", "git.exe"}:
                git_args = text.split(maxsplit=2)[1:2]
                if git_args and git_args[0].casefold() in {"status", "diff", "log", "branch"}:
                    # Git inspection has its own project-operation route;
                    # do not turn it into a generic terminal invocation.
                    return None, {}, False
            if first in self._KNOWN_COMMANDS:
                return self._prepare_terminal_request("terminal.run", text, explicit=False)
        return None, {}, False

    @staticmethod
    def _clean_path(value: str) -> str:
        return value.strip().strip("\"'").rstrip(" \t\r\n.,!?;。！？；")

    def _prepare_file_request(
        self,
        tool_name: str,
        remainder: str,
    ) -> tuple[str | None, dict[str, Any], bool]:
        value = remainder.strip()
        if tool_name in {"file.write", "file.append"}:
            marker, separator, content = value.rpartition(":")
            if not separator or not marker.strip() or not content.strip():
                return tool_name, {}, False
            path = self._clean_path(marker)
            if not path:
                return tool_name, {}, False
            return tool_name, {"path": path, "content": content.strip()}, True
        if tool_name == "file.list" and not value:
            return tool_name, {"path": "."}, True
        if not value:
            return tool_name, {}, False
        try:
            tokens = shlex.split(value, posix=False)
        except ValueError:
            tokens = []
        path = self._clean_path(tokens[0] if tokens else value.split()[0])
        if not path:
            return tool_name, {}, False
        return tool_name, {"path": path}, True

    def _prepare_terminal_request(
        self,
        tool_name: str,
        command: str,
        *,
        explicit: bool,
    ) -> tuple[str | None, dict[str, Any], bool]:
        try:
            argv = shlex.split(command, posix=False)
        except ValueError:
            return tool_name, {}, False
        argv = [item.strip("\"'") for item in argv]
        if not argv or any(item in {">", ">>", "<", "<<", "|", "||", "&&", ";"} for item in argv):
            return tool_name, {}, False
        if not explicit and Path(argv[0]).name.casefold() not in self._KNOWN_COMMANDS:
            return tool_name, {}, False
        return tool_name, {"argv": argv}, True

    def route(
        self,
        message: str,
        *,
        requested_tool_name: str | None = None,
        parsed_arguments: Mapping[str, Any] | None = None,
        arguments_validated: bool = False,
        intent_hint: str | None = None,
        classifier: Classifier | None = None,
    ) -> CapabilityRoute:
        text = message.strip()
        normalized = text.casefold()

        if requested_tool_name:
            return self._route_tool(
                requested_tool_name,
                arguments_ready=parsed_arguments is not None and arguments_validated,
                reason="explicit registered tool name",
            )

        mentioned_tools = self._mentioned_tools(normalized)
        if len(mentioned_tools) == 1:
            return self._route_tool(
                mentioned_tools[0],
                arguments_ready=parsed_arguments is not None and arguments_validated,
                reason="registered tool name in request",
            )
        if len(mentioned_tools) > 1:
            return self._clarify("tool", "multiple registered tool names require disambiguation")

        if intent_hint:
            return self._route_intent_hint(intent_hint, parsed_arguments, arguments_validated)

        # Lexical questions are knowledge requests even when they mention memory or Git.
        # A trailing question mark alone is handled after explicit operation rules.
        if self._is_question(normalized):
            return self._rule_route("knowledge_qa", "knowledge_qa", "explicit question form")

        for markers, tool_name in self._FILE_RULES:
            if self._contains(normalized, markers):
                return self._route_tool(
                    tool_name,
                    arguments_ready=parsed_arguments is not None and arguments_validated,
                    reason="explicit file operation",
                )

        if self._is_memory_operation(normalized):
            return self._rule_route("memory_operation", "memory", "explicit memory operation")
        if self._is_schedule_operation(normalized):
            return self._rule_route("project_operation", "schedule", "explicit schedule operation")
        if self._contains(normalized, self._GIT_ACTIONS):
            return self._rule_route("project_operation", "git", "explicit Git operation")
        if self._contains(normalized, self._COMPLEX_MARKERS):
            return self._rule_route("complex_task", "complex_task", "multi-step or cross-capability request")
        if "?" in normalized or "？" in normalized:
            return self._rule_route("knowledge_qa", "knowledge_qa", "question punctuation")
        if self._is_conversation(normalized):
            return CapabilityRoute(
                route="conversation",
                intent="conversation",
                confidence=1.0,
                reason="deterministic conversation marker",
            )

        if classifier is None:
            return CapabilityRoute(
                route="clarify",
                intent="ambiguous",
                confidence=0.0,
                reason="no deterministic route and no model classifier configured",
            )

        relevant = self._registry.planner_candidates_for_request(
            text,
            limit=PLANNER_CANDIDATE_LIMIT,
        )
        manifests = self._registry.manifests_for_planner_candidates(relevant)
        raw = classifier(text, manifests)
        model_route = CapabilityRoute.model_validate(raw)
        filtered = self._filter_model_route(model_route, relevant)
        if (
            filtered.route == "direct_execution"
            and filtered.confidence < self._direct_execution_confidence_threshold
        ):
            return CapabilityRoute(
                route="clarify",
                intent=filtered.intent,
                candidate_capability_ids=filtered.candidate_capability_ids,
                candidate_executor_names=filtered.candidate_executor_names,
                selected_capability_id=filtered.selected_capability_id,
                selected_executor_name=filtered.selected_executor_name,
                confidence=filtered.confidence,
                reason=(
                    "model direct-execution confidence is below configured threshold "
                    f"{self._direct_execution_confidence_threshold}"
                ),
            )
        if filtered.route == "direct_execution" and not (
            parsed_arguments is not None and arguments_validated
        ):
            return CapabilityRoute(
                route="clarify",
                intent=filtered.intent,
                candidate_capability_ids=filtered.candidate_capability_ids,
                candidate_executor_names=filtered.candidate_executor_names,
                selected_capability_id=filtered.selected_capability_id,
                selected_executor_name=filtered.selected_executor_name,
                confidence=filtered.confidence,
                reason="selected executor arguments are not validated",
            )
        return filtered

    @staticmethod
    def _contains(text: str, markers: Sequence[str]) -> bool:
        for marker in markers:
            normalized = marker.casefold()
            if re.fullmatch(r"[a-z][a-z0-9 _-]*", normalized):
                pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])"
                if re.search(pattern, text):
                    return True
            elif normalized in text:
                return True
        return False

    def _mentioned_tools(self, text: str) -> list[str]:
        matches: list[str] = []
        for manifest in self._registry.list_manifests():
            for tool_name in manifest.tool_names:
                pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(tool_name.casefold())}(?![A-Za-z0-9_.-])"
                if re.search(pattern, text):
                    matches.append(tool_name)
        return sorted(set(matches))

    def _is_conversation(self, text: str) -> bool:
        stripped = text.strip()
        if stripped in {marker.casefold() for marker in self._CONVERSATION_MARKERS}:
            return True
        return bool(re.fullmatch(r"(?:hello|hi)(?:[!,. ]+)?", stripped))

    def _is_question(self, text: str) -> bool:
        if self._contains(text, self._QUESTION_MARKERS):
            return True
        return bool(re.search(r"\b(?:why|what|how|explain)\b", text))

    def _is_memory_operation(self, text: str) -> bool:
        return self._contains(text, self._MEMORY_MARKERS) and self._contains(text, self._MEMORY_ACTIONS)

    def _is_schedule_operation(self, text: str) -> bool:
        return self._contains(text, self._SCHEDULE_MARKERS) and self._contains(text, self._SCHEDULE_ACTIONS)

    def _route_tool(self, tool_name: str, *, arguments_ready: bool, reason: str) -> CapabilityRoute:
        manifest = self._registry.find_by_tool_name(tool_name)
        if manifest is None:
            return self._clarify("tool", f"unknown tool: {tool_name}")
        if manifest.status != "available":
            return self._clarify(
                manifest.intents[0] if manifest.intents else "tool",
                f"capability {manifest.capability_id} is {manifest.status}",
            )
        route: RouteName = "direct_execution" if arguments_ready else "clarify"
        return CapabilityRoute(
            route=route,
            intent=manifest.intents[0] if manifest.intents else "tool",
            candidate_capability_ids=[manifest.capability_id],
            candidate_executor_names=[tool_name],
            selected_capability_id=manifest.capability_id,
            selected_executor_name=tool_name,
            confidence=1.0,
            reason=reason if arguments_ready else f"{reason}; arguments are not validated",
        )

    def _route_intent_hint(
        self,
        intent_hint: str,
        parsed_arguments: Mapping[str, Any] | None,
        arguments_validated: bool,
    ) -> CapabilityRoute:
        try:
            manifest = self._registry.discover(intent_hint)
        except KeyError:
            candidates = self._registry.planner_candidates_for_intents([intent_hint])
            if not candidates:
                return self._clarify(intent_hint, f"no available capability for intent: {intent_hint}")
            if len(candidates) != 1 or len(candidates[0].executor_names) != 1:
                return self._route_from_candidates(
                    "clarify",
                    intent_hint,
                    candidates,
                    "intent matches multiple executors",
                )
            return self._direct_from_candidate(
                candidates[0],
                intent_hint,
                parsed_arguments is not None and arguments_validated,
                "explicit intent hint",
            )
        if manifest.status != "available":
            if manifest.status == "interface_only":
                return CapabilityRoute(
                    route="clarify",
                    intent=intent_hint,
                    confidence=1.0,
                    reason=(
                        f"capability {manifest.capability_id} is interface_only; "
                        "no executable domain step exists"
                    ),
                    capability_gap=CapabilityGap(
                        capability_id=manifest.capability_id,
                        requested_by_step_id="route",
                        description=(
                            f"The {manifest.capability_id} interface exists but has no "
                            "verified executable implementation."
                        ),
                        required_contract={
                            "type": "object",
                            "intent": intent_hint,
                            "status": "interface_only",
                        },
                    ),
                )
            return self._clarify(intent_hint, f"capability {manifest.capability_id} is {manifest.status}")
        names = list(manifest.tool_names) or [manifest.capability_id]
        candidate = PlannerCandidate(capability_id=manifest.capability_id, executor_names=names[:12])
        if len(candidate.executor_names) != 1:
            return self._route_from_candidates(
                "clarify", intent_hint, [candidate], "capability exposes multiple executors"
            )
        return self._direct_from_candidate(
            candidate,
            intent_hint,
            parsed_arguments is not None and arguments_validated,
            "explicit capability hint",
        )

    def _rule_route(self, route: RouteName, intent: str, reason: str) -> CapabilityRoute:
        candidates = self._registry.planner_candidates_for_intents([intent])
        return self._route_from_candidates(route, intent, candidates, reason)

    @staticmethod
    def _route_from_candidates(
        route: RouteName,
        intent: str,
        candidates: Sequence[PlannerCandidate],
        reason: str,
    ) -> CapabilityRoute:
        capability_ids = [item.capability_id for item in candidates]
        executor_names = [name for item in candidates for name in item.executor_names]
        selected_capability = capability_ids[0] if len(capability_ids) == 1 else None
        return CapabilityRoute(
            route=route,
            intent=intent,
            candidate_capability_ids=capability_ids,
            candidate_executor_names=executor_names,
            selected_capability_id=selected_capability,
            confidence=1.0,
            reason=reason,
        )

    @staticmethod
    def _direct_from_candidate(
        candidate: PlannerCandidate,
        intent: str,
        arguments_ready: bool,
        reason: str,
    ) -> CapabilityRoute:
        executor_name = candidate.executor_names[0]
        return CapabilityRoute(
            route="direct_execution" if arguments_ready else "clarify",
            intent=intent,
            candidate_capability_ids=[candidate.capability_id],
            candidate_executor_names=[executor_name],
            selected_capability_id=candidate.capability_id,
            selected_executor_name=executor_name,
            confidence=1.0,
            reason=reason if arguments_ready else f"{reason}; arguments are not validated",
        )

    @staticmethod
    def _clarify(intent: str, reason: str) -> CapabilityRoute:
        return CapabilityRoute(route="clarify", intent=intent, confidence=1.0, reason=reason)

    @staticmethod
    def _filter_model_route(
        route: CapabilityRoute,
        relevant: Sequence[PlannerCandidate],
    ) -> CapabilityRoute:
        allowed_pairs = {
            item.capability_id: set(item.executor_names)
            for item in relevant
        }
        allowed_capability_ids = set(allowed_pairs)
        allowed_executor_names = {name for item in relevant for name in item.executor_names}
        requested_capability_ids = [
            item for item in route.candidate_capability_ids if item in allowed_capability_ids
        ]
        requested_executor_names = [
            item for item in route.candidate_executor_names if item in allowed_executor_names
        ]
        capability_ids = [
            capability_id
            for capability_id in requested_capability_ids
            if any(
                executor_name in allowed_pairs[capability_id]
                for executor_name in requested_executor_names
            )
        ]
        executor_names = [
            executor_name
            for executor_name in requested_executor_names
            if any(executor_name in allowed_pairs[capability_id] for capability_id in capability_ids)
        ]
        selected_capability = route.selected_capability_id
        selected_executor = route.selected_executor_name
        unavailable = (
            selected_capability is not None and selected_capability not in allowed_capability_ids
        ) or (selected_executor is not None and selected_executor not in allowed_executor_names)
        if selected_capability is not None and selected_executor is not None:
            unavailable = unavailable or selected_executor not in allowed_pairs.get(
                selected_capability, set()
            )
        if unavailable:
            selected = selected_executor or selected_capability or "unknown"
            return CapabilityRoute(
                route="clarify",
                intent=route.intent,
                candidate_capability_ids=capability_ids,
                candidate_executor_names=executor_names,
                confidence=route.confidence,
                reason=f"model selected unavailable or irrelevant executor: {selected}",
            )
        return CapabilityRoute(
            route=route.route,
            intent=route.intent,
            candidate_capability_ids=capability_ids,
            candidate_executor_names=executor_names,
            selected_capability_id=selected_capability,
            selected_executor_name=selected_executor,
            confidence=route.confidence,
            reason=route.reason,
        )


__all__ = ["CapabilityRoute", "CapabilityRouter", "Classifier", "RouteName"]
