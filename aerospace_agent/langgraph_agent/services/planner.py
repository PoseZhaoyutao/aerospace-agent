"""Protocol-constrained LLM planner for work and tool intents."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..schema import ActionType, Decision
from ..prompts import AEROSPACE_ASSISTANT_IDENTITY


def _latest_human_text(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("messages", []) or []):
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _bounded_json(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


class LLMPlanner:
    """Ask the configured model for one validated graph-routing decision."""

    def __init__(self, llm: Any, *, tool_names: Iterable[str] = ()) -> None:
        if llm is None:
            raise ValueError("LLM planner requires a model client")
        self.llm = llm
        self.tool_names = tuple(sorted({str(name) for name in tool_names if str(name)}))

    def plan(self, state: Mapping[str, Any]) -> Decision:
        question = _latest_human_text(state)
        evidence = [
            str(item.get("excerpt", ""))
            for item in state.get("evidence", []) or []
            if isinstance(item, Mapping) and item.get("excerpt")
        ][:3]
        tool_results = list(state.get("tool_results", []) or [])[-3:]
        observation = state.get("observation")
        previous_decision = state.get("decision")
        step_count = int(state.get("step_count", 0) or 0)
        prompt = (
            "Choose the next action for an aerospace work request. Return exactly one JSON object "
            "matching this contract: {\"action\":\"respond|retrieve|call_tool|stop\","
            "\"rationale\":\"...\",\"tool_request\":{\"tool_name\":\"...\","
            "\"arguments\":{}}}. Omit tool_request unless action is call_tool. "
            "Use retrieve only when private evidence is genuinely needed and has not already been supplied. "
            "Use call_tool only for an available tool. Never repeat the same successful tool call with "
            "the same arguments. Inspect execution context before choosing the next action. "
            "Do not answer the user in this decision.\n\n"
            f"Intent: {state.get('intent', 'general')}\n"
            f"Request: {question}\n"
            f"Evidence already supplied: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Step count: {step_count}\n"
            f"Previous decision: {_bounded_json(previous_decision, 1000)}\n"
            f"Recent tool results: {_bounded_json(tool_results, 3500)}\n"
            f"Observation: {_bounded_json(observation, 1500)}\n"
            f"Available tools: {json.dumps(self.tool_names, ensure_ascii=False)}"
        )
        response = self.llm.chat(
            prompt,
            system_prompt=(
                f"{AEROSPACE_ASSISTANT_IDENTITY}\n"
                "在内部执行工作规划协议：只返回符合契约的 JSON 决策，不直接回答用户。"
            ),
            max_tokens=384,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
        )
        match = re.search(r"\{.*\}", str(response or ""), re.DOTALL)
        if match is None:
            raise ValueError("planner model did not return a JSON decision")
        decision = Decision.model_validate(json.loads(match.group(0)))
        if decision.action == ActionType.CALL_TOOL:
            request = decision.tool_request
            if request is None:
                raise ValueError("call_tool decision omitted tool_request")
            if request.tool_name not in self.tool_names:
                raise ValueError(f"planner tool is not available: {request.tool_name}")
        return decision


__all__ = ["LLMPlanner"]
