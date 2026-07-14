from __future__ import annotations

import pytest
from pydantic import ValidationError

from aerospace_agent.langgraph_agent.agent_core.capabilities import CapabilityRegistry
from aerospace_agent.langgraph_agent.agent_core.models import CapabilityGap, CapabilityManifest
from aerospace_agent.langgraph_agent.agent_core.routing import CapabilityRoute, CapabilityRouter


def _manifest(
    capability_id: str,
    *,
    status: str = "available",
    intents: list[str],
    tool_names: list[str] | None = None,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_id=capability_id,
        version="1.0.0",
        category="basic" if status == "available" else "domain",
        status=status,
        intents=intents,
        tool_names=tool_names or [],
        risk_level="read_only",
        source="aerospace_agent.mcp.tools",
    )


@pytest.fixture
def router() -> CapabilityRouter:
    return CapabilityRouter(
        CapabilityRegistry(
            [
                _manifest("files", intents=["files"], tool_names=["file.read"]),
                _manifest("memory", intents=["memory"]),
                _manifest("git", intents=["git", "project"]),
                _manifest("scheduler", intents=["schedule", "project"]),
                _manifest("simulation", status="interface_only", intents=["simulation"]),
            ]
        )
    )


def test_route_contract_is_strict_and_limits_candidates() -> None:
    with pytest.raises(ValidationError):
        CapabilityRoute(
            route="direct_execution",
            intent="files",
            candidate_capability_ids=[f"c-{index}" for index in range(13)],
            selected_capability_id=None,
            confidence=0.8,
            reason="too many",
        )

    with pytest.raises(ValidationError):
        CapabilityRoute(
            route="conversation",
            intent="conversation",
            candidate_capability_ids=[],
            selected_capability_id=None,
            confidence=1.1,
            reason="invalid confidence",
        )


def test_explicit_tool_name_routes_directly_without_model(router: CapabilityRouter) -> None:
    def classifier(_: str, __: list[CapabilityManifest]) -> dict[str, object]:
        raise AssertionError("deterministic request must not call model classifier")

    route = router.route(
        "读取文件",
        requested_tool_name="file.read",
        parsed_arguments={"path": "README.md"},
        arguments_validated=True,
        classifier=classifier,
    )

    assert route.route == "direct_execution"
    assert route.selected_capability_id == "files"
    assert route.candidate_capability_ids == ["files"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("请清空当前会话记忆", "memory_operation"),
        ("请查看 git status", "project_operation"),
        ("创建一个内部定时任务", "project_operation"),
        ("分解这个跨领域复杂任务并规划步骤", "complex_task"),
        ("为什么轨道会发生进动？", "knowledge_qa"),
        ("你好", "conversation"),
    ],
)
def test_rule_first_routes_common_requests(
    router: CapabilityRouter, message: str, expected: str
) -> None:
    route = router.route(message)

    assert route.route == expected


@pytest.mark.parametrize(
    "message",
    [
        "请规划一个两体轨道传播工作流",
        "plan a two-body orbit propagation workflow",
    ],
)
def test_domain_workflow_requests_route_to_complex_task(message: str, router: CapabilityRouter) -> None:
    route = router.route(message)

    assert route.route == "complex_task"
    assert route.intent == "complex_task"
    assert len(route.candidate_capability_ids) <= 12


def test_interface_only_capability_is_never_an_execution_candidate(
    router: CapabilityRouter,
) -> None:
    route = router.route("请运行仿真", intent_hint="simulation")

    assert route.route == "clarify"
    assert route.candidate_capability_ids == []
    assert route.selected_capability_id is None
    assert "interface_only" in route.reason


def test_explicit_file_request_is_rule_routed_but_missing_parameters_block_execution(
    router: CapabilityRouter,
) -> None:
    incomplete = router.route("调用 file.read 读取 README.md")
    complete = router.route(
        "调用 file.read 读取 README.md",
        parsed_arguments={"path": "README.md"},
        arguments_validated=True,
    )

    assert incomplete.route == "clarify"
    assert incomplete.selected_capability_id == "files"
    assert complete.route == "direct_execution"
    assert complete.selected_executor_name == "file.read"


def test_natural_file_request_prepares_validated_path_for_direct_execution(
    router: CapabilityRouter,
) -> None:
    tool_name, arguments, validated = router.prepare_request("read file AGENTS.md")

    route = router.route(
        "read file AGENTS.md",
        requested_tool_name=tool_name,
        parsed_arguments=arguments,
        arguments_validated=validated,
    )

    assert tool_name == "file.read"
    assert arguments == {"path": "AGENTS.md"}
    assert validated is True
    assert route.route == "direct_execution"
    assert route.selected_executor_name == "file.read"


def test_natural_terminal_request_prepares_argv_without_shell_operators(
    router: CapabilityRouter,
) -> None:
    tool_name, arguments, validated = router.prepare_request(
        "run command python --version"
    )

    assert tool_name == "terminal.run"
    assert arguments == {"argv": ["python", "--version"]}
    assert validated is True


def test_natural_write_request_prepares_arguments_but_does_not_bypass_confirmation(
    router: CapabilityRouter,
) -> None:
    router = CapabilityRouter(
        CapabilityRegistry(
            [_manifest("files", intents=["files"], tool_names=["file.write"])]
        )
    )
    tool_name, arguments, validated = router.prepare_request(
        "write file notes.txt: use SI units"
    )
    route = router.route(
        "write file notes.txt: use SI units",
        requested_tool_name=tool_name,
        parsed_arguments=arguments,
        arguments_validated=validated,
    )

    assert tool_name == "file.write"
    assert arguments == {"path": "notes.txt", "content": "use SI units"}
    assert validated is True
    # Routing can select the write executor; the execution boundary still
    # returns confirmation_required before any mutation.
    assert route.route == "direct_execution"


@pytest.mark.parametrize("message", ["为什么 git 会出现 detached HEAD？", "请解释 memory isolation"])
def test_knowledge_questions_are_not_misrouted_as_operations(
    router: CapabilityRouter, message: str
) -> None:
    assert router.route(message).route == "knowledge_qa"


def test_ambiguous_model_output_is_validated_and_unavailable_candidates_are_filtered(
    router: CapabilityRouter,
) -> None:
    def classifier(_: str, __: list[CapabilityManifest]) -> dict[str, object]:
        return {
            "route": "direct_execution",
            "intent": "simulation",
            "candidate_capability_ids": ["simulation"],
            "candidate_executor_names": ["simulation"],
            "selected_capability_id": "simulation",
            "selected_executor_name": "simulation",
            "confidence": 0.9,
            "reason": "model suggestion",
        }

    route = router.route("处理一下", classifier=classifier)

    assert route.route == "clarify"
    assert route.candidate_capability_ids == []
    assert route.selected_capability_id is None
    assert "unavailable" in route.reason


def test_model_is_used_only_for_ambiguous_requests(router: CapabilityRouter) -> None:
    calls: list[str] = []

    def classifier(message: str, candidates: list[CapabilityManifest]) -> dict[str, object]:
        calls.append(message)
        assert all(item.status == "available" for item in candidates)
        assert len(candidates) <= 12
        return {
            "route": "conversation",
            "intent": "conversation",
            "candidate_capability_ids": [],
            "candidate_executor_names": [],
            "selected_capability_id": None,
            "selected_executor_name": None,
            "confidence": 0.7,
            "reason": "classified ambiguous request",
        }

    route = router.route("处理一下", classifier=classifier)

    assert route.route == "conversation"
    assert calls == ["处理一下"]


def test_model_receives_request_relevant_candidate_even_if_registered_thirteenth() -> None:
    manifests = [
        _manifest(f"unrelated-{index}", intents=[f"unrelated-{index}"])
        for index in range(12)
    ]
    manifests.append(_manifest("target", intents=["target-intent"], tool_names=["target.run"]))
    router = CapabilityRouter(CapabilityRegistry(manifests))

    def classifier(_: str, candidates: list[CapabilityManifest]) -> dict[str, object]:
        assert [item.capability_id for item in candidates] == ["target"]
        return {
            "route": "direct_execution",
            "intent": "target-intent",
            "candidate_capability_ids": ["target"],
            "candidate_executor_names": ["target.run"],
            "selected_capability_id": "target",
            "selected_executor_name": "target.run",
            "confidence": 0.9,
            "reason": "relevant candidate",
        }

    route = router.route(
        "请处理 target-intent",
        parsed_arguments={},
        arguments_validated=True,
        classifier=classifier,
    )
    assert route.selected_capability_id == "target"


def test_intent_hint_with_multiple_capabilities_requires_disambiguation() -> None:
    router = CapabilityRouter(
        CapabilityRegistry(
            [
                _manifest("one", intents=["files"], tool_names=["file.one"]),
                _manifest("two", intents=["files"], tool_names=["file.two"]),
            ]
        )
    )

    route = router.route("读取", intent_hint="files")

    assert route.route == "clarify"
    assert route.selected_capability_id is None


def test_model_cannot_cross_pair_capability_and_executor() -> None:
    router = CapabilityRouter(
        CapabilityRegistry(
            [
                _manifest("cap-a", intents=["mix"], tool_names=["tool.a"]),
                _manifest("cap-b", intents=["mix"], tool_names=["tool.b"]),
            ]
        )
    )

    def classifier(_: str, __: list[CapabilityManifest]) -> dict[str, object]:
        return {
            "route": "direct_execution",
            "intent": "mix",
            "candidate_capability_ids": ["cap-a", "cap-b"],
            "candidate_executor_names": ["tool.a", "tool.b"],
            "selected_capability_id": "cap-a",
            "selected_executor_name": "tool.b",
            "confidence": 0.9,
            "reason": "invalid cross pair",
        }

    route = router.route(
        "mix",
        parsed_arguments={},
        arguments_validated=True,
        classifier=classifier,
    )

    assert route.route == "clarify"
    assert route.selected_capability_id is None


def test_tool_name_prefix_collision_uses_complete_identifier() -> None:
    router = CapabilityRouter(
        CapabilityRegistry(
            [
                _manifest(
                    "files",
                    intents=["files"],
                    tool_names=["file.read", "file.read_lines"],
                )
            ]
        )
    )

    route = router.route(
        "调用 file.read_lines",
        parsed_arguments={"path": "README.md", "start": 1},
        arguments_validated=True,
    )

    assert route.route == "direct_execution"
    assert route.selected_executor_name == "file.read_lines"


@pytest.mark.parametrize("message", ["execute this", "something"])
def test_short_greeting_substrings_do_not_force_conversation(message: str, router: CapabilityRouter) -> None:
    assert router.route(message).route == "clarify"


def test_ambiguous_request_without_classifier_fails_closed(router: CapabilityRouter) -> None:
    route = router.route("处理一下")

    assert route.route == "clarify"
    assert route.confidence == 0.0


def test_model_candidate_capability_and_executor_sets_must_have_valid_ownership() -> None:
    router = CapabilityRouter(
        CapabilityRegistry(
            [
                _manifest("cap-a", intents=["mix"], tool_names=["tool.a"]),
                _manifest("cap-b", intents=["mix"], tool_names=["tool.b"]),
            ]
        )
    )

    def classifier(_: str, __: list[CapabilityManifest]) -> dict[str, object]:
        return {
            "route": "complex_task",
            "intent": "mix",
            "candidate_capability_ids": ["cap-a"],
            "candidate_executor_names": ["tool.b"],
            "selected_capability_id": None,
            "selected_executor_name": None,
            "confidence": 0.8,
            "reason": "mismatched candidates",
        }

    route = router.route("mix", classifier=classifier)

    assert route.candidate_capability_ids == []
    assert route.candidate_executor_names == []


def test_english_question_words_use_boundaries_before_file_rules(router: CapabilityRouter) -> None:
    route = router.route(
        "show file README.md",
        parsed_arguments={"path": "README.md"},
        arguments_validated=True,
    )

    assert route.route == "direct_execution"
    assert route.selected_executor_name == "file.read"


def test_interface_only_intent_returns_structured_gap_without_execution_candidate(
    router: CapabilityRouter,
) -> None:
    route = router.route(
        "run the simulation domain",
        intent_hint="simulation",
        parsed_arguments={"state": {}},
        arguments_validated=True,
    )

    assert route.route == "clarify"
    assert isinstance(route.capability_gap, CapabilityGap)
    assert route.capability_gap.capability_id == "simulation"
    assert route.capability_gap.requested_by_step_id == "route"
    assert route.candidate_capability_ids == []
    assert route.candidate_executor_names == []
    assert route.selected_capability_id is None
    assert route.selected_executor_name is None


def test_trailing_question_mark_does_not_override_explicit_file_or_git_operation(
    router: CapabilityRouter,
) -> None:
    file_route = router.route(
        "请读取 README.md？",
        parsed_arguments={"path": "README.md"},
        arguments_validated=True,
    )
    git_route = router.route("请查看 git status？")

    assert file_route.route == "direct_execution"
    assert git_route.route == "project_operation"


@pytest.mark.parametrize(
    "message",
    ["memory clearance policy", "scheduler runtime design", "summarize appendix A"],
)
def test_english_operation_markers_use_word_boundaries(
    router: CapabilityRouter,
    message: str,
) -> None:
    assert router.route(message).route == "clarify"


def test_low_confidence_model_direct_execution_is_downgraded_to_clarify() -> None:
    router = CapabilityRouter(
        CapabilityRegistry([_manifest("files", intents=["files"], tool_names=["file.read"])])
    )

    def classifier(_: str, __: list[CapabilityManifest]) -> dict[str, object]:
        return {
            "route": "direct_execution",
            "intent": "files",
            "candidate_capability_ids": ["files"],
            "candidate_executor_names": ["file.read"],
            "selected_capability_id": "files",
            "selected_executor_name": "file.read",
            "confidence": 0.0,
            "reason": "uncertain",
        }

    route = router.route(
        "files",
        parsed_arguments={"path": "README.md"},
        arguments_validated=True,
        classifier=classifier,
    )

    assert route.route == "clarify"
    assert "confidence" in route.reason


@pytest.mark.parametrize("route_name", ["conversation", "knowledge_qa"])
def test_non_execution_routes_cannot_select_an_executor(route_name: str) -> None:
    with pytest.raises(ValidationError, match="must not select"):
        CapabilityRoute(
            route=route_name,
            intent="files",
            candidate_capability_ids=["files"],
            candidate_executor_names=["file.read"],
            selected_capability_id="files",
            selected_executor_name="file.read",
            confidence=0.9,
            reason="contradictory selection",
        )

