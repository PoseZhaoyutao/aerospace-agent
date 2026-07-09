"""标准 ReAct Agent 实现 — 基于 langchain-core 架构。

特性:
    - 结构化: AgentAction / AgentFinish 明确区分
    - 上下文管理: Token 预算强制执行（Phase 1~4 渐进式压缩）
    - 流式输出: stream() 方法支持实时流式输出
    - 错误恢复: 错误计数 → 提示换工具 → 阈值终止
    - 重复检测: 相同工具+相同参数直接终止
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool

from .react_parser import (
    AgentAction,
    AgentFinish,
    ReActOutputParser,
)


@dataclass
class AgentConfig:
    """ReAct Agent 配置。"""
    max_steps: int = 20
    max_consecutive_errors: int = 2
    max_repeated_actions: int = 2
    system_prompt: Optional[str] = None
    stream_callback: Optional[Callable[[str], None]] = None
    max_context_tokens: int = 8192
    max_output_tokens: int = 1024


@dataclass
class AgentResult:
    """Agent 执行结果。"""
    output: str
    steps: int
    errors: int
    early_termination: bool = False
    termination_reason: str = ""


class ReActAgent:
    """标准 ReAct Agent。"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        parser: Optional[ReActOutputParser] = None,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_names = list(self.tools.keys())
        self.parser = parser or ReActOutputParser()

    @staticmethod
    def _estimate_messages_tokens(messages: List[BaseMessage]) -> int:
        """估算消息列表的 token 数（保守估计）。"""
        total = 0
        for m in messages:
            content = m.content if hasattr(m, "content") else str(m)
            if isinstance(content, str):
                total += len(content) // 3
            else:
                total += len(str(content)) // 3
        return int(total * 4 / 3)

    def _enforce_token_budget(
        self,
        messages: List[BaseMessage],
        max_tokens: int,
        output_tokens: int,
    ) -> List[BaseMessage]:
        """强制执行 token 预算——超限时渐进式压缩。

        四级策略（按破坏性从低到高）:
          Phase 1: 清除旧 Observation/ToolMessage 内容（保留最近 2 条）
          Phase 2: 截断过长单条消息（>7000 字符）
          Phase 3: 丢弃最旧非系统消息（保留最近 6 条）
          Phase 4: 截断剩余 Observation 到 500 字符

        SystemMessage 永不丢弃。
        """
        safe_budget = max(512, max_tokens - output_tokens)

        current = self._estimate_messages_tokens(messages)
        if current <= safe_budget:
            return messages

        # 分离 system 和非 system
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # Phase 1: 清除旧 Observation
        obs_indices = [
            i for i, m in enumerate(non_system)
            if (isinstance(m, ToolMessage)
                or (isinstance(m, HumanMessage) and "Observation:" in str(m.content)))
        ]
        if len(obs_indices) > 2:
            for i in obs_indices[:-2]:
                non_system[i] = HumanMessage(content="[旧工具结果已清除以节省上下文空间]")

        current = self._estimate_messages_tokens(system_msgs + non_system)
        if current <= safe_budget:
            return system_msgs + non_system

        # Phase 2: 截断超长消息
        for i, m in enumerate(non_system):
            content = str(m.content) if hasattr(m, "content") else str(m)
            if len(content) > 7000:
                non_system[i] = HumanMessage(
                    content=content[:7000]
                    + f"\n...[已截断，原长度 {len(content)} 字符]..."
                )

        current = self._estimate_messages_tokens(system_msgs + non_system)
        if current <= safe_budget:
            return system_msgs + non_system

        # Phase 3: 丢弃最旧非系统消息（保留最近 6 条）
        min_keep = 6
        while (
            len(non_system) > min_keep
            and self._estimate_messages_tokens(system_msgs + non_system) > safe_budget
        ):
            non_system.pop(0)

        # Phase 4: 仍然超标——截断剩余 Observation 到 500 字符
        current = self._estimate_messages_tokens(system_msgs + non_system)
        if current > safe_budget:
            for i, m in enumerate(non_system):
                content = str(m.content) if hasattr(m, "content") else str(m)
                if "Observation:" in content and len(content) > 500:
                    non_system[i] = HumanMessage(content=content[:500] + "...[紧急截断]")

        # 如果删除后第一条是 assistant，插入 user 占位符保持角色交替
        if non_system and isinstance(non_system[0], AIMessage):
            non_system.insert(0, HumanMessage(content="[更早的对话已压缩]"))

        return system_msgs + non_system

    def build_system_prompt(self, custom_system: Optional[str] = None) -> str:
        """构建标准 ReAct system prompt + 工具列表。"""
        base = """你是航天动力学 Agent，使用 ReAct 模式解决航天任务问题。

回复必须严格遵循以下格式之一：

格式 1 (调用工具):
Thought: <你的推理，说明为什么调用这个工具>
Action: <工具名>
Action Input: <JSON 参数对象>

格式 2 (返回答案):
Thought: <总结推理>
Final Answer: <最终答案文本>

可用工具（按名称列出）：
"""
        tools_section = "\n".join(
            f"  - **{tool.name}**: {tool.description}"
            for tool in self.tools.values()
        )
        prompt = base + tools_section + """

规则：
- 如果已经有了足够信息回答任务，直接输出 Final Answer，不要调用工具。
- 如果需要工具帮助，严格按照格式输出 Action + Action Input。
- 工具调用失败后，你必须立即换用其他工具或调整参数，不得重复调用同一个错误工具。
- JSON 参数必须是有效的 JSON 格式。
"""
        if custom_system:
            prompt = custom_system + "\n\n" + prompt
        return prompt

    def _action_signature(self, action: AgentAction) -> str:
        """生成动作签名，用于检测重复调用。"""
        try:
            args_text = str(sorted(action.tool_input.items()))
        except Exception:
            args_text = str(action.tool_input)
        return f"{action.tool}:{args_text}"

    def _call_llm(
        self,
        messages: List[BaseMessage],
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """调用 LLM，可选流式输出。"""
        if stream_callback and hasattr(self.llm, "stream_chat"):
            # 流式模式：使用底层 LLM 的 stream_chat
            from ..llm_interface import normalize_messages_for_api
            converted = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    converted.append({"role": "system", "content": str(m.content)})
                elif isinstance(m, HumanMessage):
                    converted.append({"role": "user", "content": str(m.content)})
                elif isinstance(m, AIMessage):
                    converted.append({"role": "assistant", "content": str(m.content)})
                else:
                    converted.append({"role": "user", "content": str(m.content)})
            chunks = []
            for chunk in self.llm.stream_chat(converted):
                stream_callback(chunk)
                chunks.append(chunk)
            return "".join(chunks)
        else:
            # 非流式模式
            response = self.llm.generate([messages])
            if response.generations and response.generations[0]:
                return response.generations[0][0].text
            return ""

    def _run_step(
        self,
        messages: List[BaseMessage],
        cfg: AgentConfig,
        consecutive_errors: int,
        repeated_actions: Dict[str, int],
    ) -> tuple[bool, AgentResult, List[BaseMessage], int, Dict[str, int]]:
        """执行单步推理。

        Returns:
            (done, result_or_none, new_messages, new_errors, new_repeated)
        """
        text = self._call_llm(messages, cfg.stream_callback)
        if not text.strip():
            consecutive_errors += 1
            if consecutive_errors >= cfg.max_consecutive_errors:
                return True, AgentResult(
                    output="连续多次空响应，终止推理。",
                    steps=0, errors=1, early_termination=True,
                    termination_reason="empty_response",
                ), messages, 1, repeated_actions
            messages.append(AIMessage(content="(空响应)"))
            messages.append(HumanMessage(content="提示: 请重新生成，请不要输出空响应。"))
            return False, None, messages, 1, repeated_actions

        parse_result = self.parser.parse(text)

        if parse_result.error:
            consecutive_errors += 1
            if consecutive_errors >= cfg.max_consecutive_errors:
                return True, AgentResult(
                    output=f"连续 {cfg.max_consecutive_errors} 次解析错误，终止推理。\n"
                           f"最后错误: {parse_result.error}",
                    steps=0, errors=1, early_termination=True,
                    termination_reason="parse_error",
                ), messages, 1, repeated_actions
            messages.append(AIMessage(content=text))
            messages.append(HumanMessage(content=(
                f"[系统提示] 输出格式解析失败: {parse_result.error}\n"
                "你必须严格遵守格式:\n"
                "  Thought: ...\n  Action: toolname\n  Action Input: {...}\n"
                "  或者\n  Final Answer: ...\n请重新输出正确格式。"
            )))
            return False, None, messages, 1, repeated_actions

        if parse_result.finish:
            output = parse_result.finish.return_values.get("output", "")
            return True, AgentResult(
                output=str(output), steps=0, errors=0,
            ), messages, 0, repeated_actions

        if parse_result.action:
            action = parse_result.action
            sig = self._action_signature(action)
            cnt = repeated_actions.get(sig, 0) + 1
            repeated_actions[sig] = cnt
            if cnt >= cfg.max_repeated_actions:
                return True, AgentResult(
                    output=f"重复调用同一动作 {cnt} 次，终止推理。\n"
                           f"工具: {action.tool}\n参数: {action.tool_input}",
                    steps=0, errors=0, early_termination=True,
                    termination_reason="repeated_action",
                ), messages, 0, repeated_actions

            tool = self.tools.get(action.tool)
            if tool is None:
                tool_list = ", ".join(self.tool_names[:10])
                if len(self.tool_names) > 10:
                    tool_list += f" ...等 {len(self.tool_names)} 个工具"
                error_msg = (
                    f"[TOOL ERROR] 工具 '{action.tool}' 不存在。\n"
                    f"可用工具: {tool_list}\n"
                    f"提示: 你必须换用其他工具，或调用 list_tools 查看完整列表。"
                )
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=error_msg))
                return False, None, messages, 1, repeated_actions

            try:
                result = tool.invoke(action.tool_input)
            except Exception as e:
                result = f"[TOOL ERROR] 工具 '{action.tool}' 执行异常: {e}"

            messages.append(AIMessage(content=text))
            messages.append(ToolMessage(content=str(result), tool_call_id=sig))

            if "[TOOL ERROR]" in str(result) or "错误" in str(result):
                if consecutive_errors + 1 >= cfg.max_consecutive_errors:
                    return True, AgentResult(
                        output=f"连续 {cfg.max_consecutive_errors} 次工具错误，终止推理。\n"
                               f"最后结果: {result}",
                        steps=0, errors=1, early_termination=True,
                        termination_reason="tool_error",
                    ), messages, 1, repeated_actions
                error_hint = (
                    f"[系统提示] 上一步工具调用失败（第 {consecutive_errors + 1} 次）。\n"
                    f"错误详情:\n{result}\n\n"
                    f"你必须立即换用其他工具或调整参数。再次失败将被强制终止。"
                )
                messages.append(HumanMessage(content=error_hint))
                return False, None, messages, 1, repeated_actions
            else:
                # 成功
                if cfg.stream_callback:
                    cfg.stream_callback(f"\n[Observation] {str(result)[:200]}...\n")
                return False, None, messages, 0, repeated_actions

        return True, AgentResult(
            output="未知状态", steps=0, errors=0,
            early_termination=True, termination_reason="unknown",
        ), messages, 0, repeated_actions

    def invoke(self, task: str, config: Optional[AgentConfig] = None) -> AgentResult:
        """同步执行 Agent 直到得出结果。"""
        cfg = config or AgentConfig()
        messages = self._init_messages(task, cfg.system_prompt)
        consecutive_errors = 0
        repeated_actions: Dict[str, int] = {}
        errors = 0

        for step in range(1, cfg.max_steps + 1):
            # Token 预算强制执行
            pre_tokens = self._estimate_messages_tokens(messages)
            messages = self._enforce_token_budget(
                messages,
                max_tokens=cfg.max_context_tokens,
                output_tokens=cfg.max_output_tokens,
            )
            post_tokens = self._estimate_messages_tokens(messages)
            if post_tokens < pre_tokens:
                # 通知流式回调上下文已压缩
                if cfg.stream_callback:
                    cfg.stream_callback(
                        f"\n[Context Compacted] {pre_tokens} -> {post_tokens} tokens\n"
                    )

            done, result, messages, step_errors, repeated_actions = self._run_step(
                messages, cfg, consecutive_errors, repeated_actions
            )
            errors += step_errors
            if step_errors > 0:
                consecutive_errors += step_errors
            else:
                consecutive_errors = 0

            if done and result is not None:
                result.steps = step
                result.errors = errors
                return result

        return AgentResult(
            output=f"已达到最大推理步数 {cfg.max_steps}，未得出最终答案。",
            steps=cfg.max_steps, errors=errors,
            early_termination=True, termination_reason="max_steps",
        )

    def stream(
        self,
        task: str,
        config: Optional[AgentConfig] = None,
    ) -> Iterator[str]:
        """流式执行 Agent，每步/每块实时输出。

        Yields:
            每个流式块（LLM 输出 chunk 或 Observation 摘要）
        """
        cfg = config or AgentConfig()
        messages = self._init_messages(task, cfg.system_prompt)
        consecutive_errors = 0
        repeated_actions: Dict[str, int] = {}

        for step in range(1, cfg.max_steps + 1):
            # Token 预算强制执行
            messages = self._enforce_token_budget(
                messages,
                max_tokens=cfg.max_context_tokens,
                output_tokens=cfg.max_output_tokens,
            )

            # 流式 LLM 调用
            text = self._call_llm(messages, lambda chunk: None)
            if not text.strip():
                consecutive_errors += 1
                if consecutive_errors >= cfg.max_consecutive_errors:
                    yield "\n[系统] 连续多次空响应，终止推理。\n"
                    return
                messages.append(AIMessage(content="(空响应)"))
                messages.append(HumanMessage(content="提示: 请重新生成。"))
                continue

            yield text

            parse_result = self.parser.parse(text)

            if parse_result.finish:
                yield f"\n[Final Answer] {parse_result.finish.return_values.get('output', '')}\n"
                return

            if parse_result.error:
                consecutive_errors += 1
                if consecutive_errors >= cfg.max_consecutive_errors:
                    yield f"\n[系统] 连续解析错误，终止。\n"
                    return
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=f"解析失败: {parse_result.error}"))
                continue

            if parse_result.action:
                action = parse_result.action
                sig = self._action_signature(action)
                cnt = repeated_actions.get(sig, 0) + 1
                repeated_actions[sig] = cnt
                if cnt >= cfg.max_repeated_actions:
                    yield f"\n[系统] 重复调用 {cnt} 次，终止。\n"
                    return

                tool = self.tools.get(action.tool)
                if tool is None:
                    yield f"\n[TOOL ERROR] 工具 '{action.tool}' 不存在\n"
                    messages.append(AIMessage(content=text))
                    messages.append(HumanMessage(content=f"[系统] 工具 '{action.tool}' 不存在。"))
                    consecutive_errors += 1
                    continue

                try:
                    result = tool.invoke(action.tool_input)
                except Exception as e:
                    result = f"[TOOL ERROR] {e}"

                messages.append(AIMessage(content=text))
                messages.append(ToolMessage(content=str(result), tool_call_id=sig))
                yield f"\n[Observation] {str(result)[:200]}...\n"

                if "[TOOL ERROR]" in str(result):
                    consecutive_errors += 1
                    if consecutive_errors >= cfg.max_consecutive_errors:
                        yield f"\n[系统] 连续 {cfg.max_consecutive_errors} 次错误，终止。\n"
                        return
                    messages.append(HumanMessage(content=(
                        f"[系统提示] 工具调用失败（第 {consecutive_errors} 次）。"
                        f"请换用其他工具。"
                    )))
                else:
                    consecutive_errors = 0

        yield f"\n[系统] 达到最大步数 {cfg.max_steps}。\n"

    def _init_messages(
        self,
        task: str,
        custom_system: Optional[str] = None,
    ) -> List[BaseMessage]:
        system_prompt = self.build_system_prompt(custom_system)
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task),
        ]