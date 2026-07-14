from __future__ import annotations

from datetime import UTC, datetime

from aerospace_agent.langgraph_agent.agent_core.capabilities import CapabilityRegistry
from aerospace_agent.langgraph_agent.agent_core.models import CapabilityManifest
from aerospace_agent.langgraph_agent.services.task_planning import (
    DeterministicReviewAssessor,
    LLMTaskPlanService,
)


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.response


class _ExecutionRegistry:
    def snapshot(self, capability_id: str):
        from aerospace_agent.langgraph_agent.agent_core.models import CapabilitySnapshot

        return CapabilitySnapshot(
            capability_id=capability_id,
            version="1.0.0",
            manifest_sha256="1" * 64,
            adapter_sha256="2" * 64,
        )


def _registry() -> CapabilityRegistry:
    return CapabilityRegistry(
        [
            CapabilityManifest(
                capability_id="basic.files",
                version="1.0.0",
                category="basic",
                status="available",
                intents=["file", "terminal"],
                tool_names=["file.read", "terminal.run"],
                risk_level="read_only",
                source="aerospace_agent.mcp.tools.core_tool_adapters",
            )
        ]
    )


def test_llm_task_plan_service_calls_model_and_builds_bound_immutable_plan():
    llm = _FakeLLM(
        '{"steps":[{"tool_name":"file.read","inputs":{"path":"AGENTS.md"}},'
        '{"tool_name":"terminal.run","inputs":{"argv":["python","--version"]},'
        '"dependencies":["step-1"]}]}'
    )
    service = LLMTaskPlanService(llm, _registry(), _ExecutionRegistry())

    result = service.create_task_plan(
        route=None,
        state={
            "project_id": "project-1",
            "thread_id": "thread-1",
            "root_run_id": "run-1",
            "messages": [],
        },
    )

    plan = result["task_plan"]
    assert plan.project_id == "project-1"
    assert [step.tool_name for step in plan.steps] == ["file.read", "terminal.run"]
    assert plan.steps[1].dependencies == ["step-1"]
    assert llm.calls
    assert "file.read" in str(llm.calls[0]["prompt"])


def test_review_assessor_binds_required_checks_to_tool_audits():
    assessor = DeterministicReviewAssessor()
    assessment = assessor.assess(
        plan=type(
            "Plan",
            (),
            {
                "steps": [
                    type(
                        "Step",
                        (),
                        {
                            "step_id": "step-1",
                            "verification": [
                                type(
                                    "Check",
                                    (),
                                    {
                                        "check_id": "check-1",
                                        "description": "completed",
                                        "method": "tool",
                                        "required": True,
                                        "acceptance_rule": "success",
                                    },
                                )()
                            ],
                        },
                    )()
                ]
            },
        )(),
        outcome=type(
            "Outcome",
            (),
            {
                "step_results": {
                    "step-1": type(
                        "Result",
                        (),
                        {"status": "success", "audit_id": "audit-1"},
                    )()
                }
            },
        )(),
        evidence=[],
    )

    assert assessment.goal_satisfied is True
    assert assessment.checks[0].check_id == "check-1"
    assert assessment.checks[0].evidence_refs == ["audit-1"]
