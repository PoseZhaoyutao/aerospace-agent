"""ReAct 输出解析器 — 解析 Qwen3 的 Thought/Action/Final Answer 文本格式。

Qwen3 不支持原生 function calling，工具调用通过 system prompt 注入定义，
模型以文本格式输出:
    Thought: <推理>
    Action: <工具名>
    Action Input: <JSON 参数>

    或

    Thought: <推理>
    Final Answer: <最终答案>

解析器从 LLM 文本输出中提取 AgentAction 或 AgentFinish。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentAction:
    """Agent 决定执行一个工具调用。

    Attributes:
        tool: 工具名称
        tool_input: 工具参数 (dict)
        log: 完整的 LLM 输出文本
    """
    tool: str
    tool_input: Dict[str, Any]
    log: str = ""


@dataclass
class AgentFinish:
    """Agent 决定返回最终答案。

    Attributes:
        return_values: 最终返回值 {"output": <答案>}
        log: 完整的 LLM 输出文本
    """
    return_values: Dict[str, Any]
    log: str = ""


@dataclass
class ParseResult:
    """解析结果 — 成功返回 AgentAction/AgentFinish，失败返回错误信息。

    Attributes:
        action: AgentAction（如果解析到工具调用）
        finish: AgentFinish（如果解析到 Final Answer）
        error: 错误信息（解析失败时）
    """
    action: Optional[AgentAction] = None
    finish: Optional[AgentFinish] = None
    error: Optional[str] = None


class ReActOutputParser:
    """解析 LLM 文本输出中的 ReAct 格式。

    解析规则:
        1. 查找 Final Answer: 优先，直接返回 AgentFinish
        2. 查找 Action: + Action Input: 配对，返回 AgentAction
        3. 无匹配：返回 error
    """

    # 正则：匹配 Action/Action Input 对
    ACTION_RE = re.compile(
        r"Action:\s*(.+?)\s*\n\s*Action\s*Input:\s*(.+?)(?:\n\s*(?:Thought|Action|Final|Observation)|$)",
        re.IGNORECASE | re.DOTALL,
    )

    # 正则：匹配 Final Answer
    FINAL_RE = re.compile(
        r"Final\s*Answer:\s*(.+?)$",
        re.IGNORECASE | re.DOTALL,
    )

    # 正则：匹配 Thought
    THOUGHT_RE = re.compile(
        r"Thought:\s*(.+?)(?:\n\s*(?:Action|Final|$))",
        re.IGNORECASE | re.DOTALL,
    )

    def parse(self, text: str) -> ParseResult:
        """解析 LLM 输出文本。

        Args:
            text: LLM 原始输出

        Returns:
            ParseResult — 包含 action/finish/error
        """
        if not text or not text.strip():
            return ParseResult(error="LLM 返回空响应")

        # 1. 优先匹配 Final Answer
        final_match = self.FINAL_RE.search(text)
        if final_match:
            answer = final_match.group(1).strip()
            # 如果 Final Answer 之前有 Action，忽略 Final Answer
            # （Final Answer 必须出现在独立的回合中）
            action_match = self.ACTION_RE.search(text)
            if action_match:
                # 检查 Action 和 Final Answer 的位置
                # 如果 Action 在 Final Answer 之前，这是一个工具调用回合
                act_start = action_match.start()
                fin_start = final_match.start()
                if act_start < fin_start:
                    # 这是一个带 Action 的回合，不是 Final
                    return self._parse_action(text, action_match)

            return ParseResult(
                finish=AgentFinish(
                    return_values={"output": answer},
                    log=text,
                )
            )

        # 2. 匹配 Action
        action_match = self.ACTION_RE.search(text)
        if action_match:
            return self._parse_action(text, action_match)

        # 3. 无匹配
        return ParseResult(error="无法解析输出: 未找到 Action 或 Final Answer")

    def _parse_action(self, text: str, match: re.Match) -> ParseResult:
        """从匹配结果中提取 AgentAction。"""
        tool_name = match.group(1).strip()
        args_raw = match.group(2).strip()

        # 解析参数
        args = self._parse_json_args(args_raw)
        if args is None:
            return ParseResult(
                error=f"工具 '{tool_name}' 的参数不是有效 JSON: {args_raw[:200]}"
            )

        return ParseResult(
            action=AgentAction(
                tool=tool_name,
                tool_input=args,
                log=text,
            )
        )

    @staticmethod
    def _parse_json_args(raw: str) -> Optional[Dict[str, Any]]:
        """解析 JSON 参数（多层回退）。"""
        raw = raw.strip()

        # 尝试直接 JSON 解析
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
            return {"input": result}
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试提取 JSON 块（可能有包裹文本）
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if isinstance(result, dict):
                    return result
                return {"input": result}
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试 ast.literal_eval
        try:
            import ast
            result = ast.literal_eval(raw)
            if isinstance(result, dict):
                return result
            return {"input": result}
        except (ValueError, SyntaxError):
            pass

        return None

    @staticmethod
    def extract_thought(text: str) -> str:
        """提取 Thought 文本（用于日志/流式输出）。"""
        match = ReActOutputParser.THOUGHT_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()[:200]