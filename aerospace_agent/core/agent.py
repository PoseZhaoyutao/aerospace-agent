"""主编排器：ReAct 循环 + 工具/工作流编排。

``AerospaceAgent`` 组合 llm + context_manager + memory + tools + workflows，
实现 ReAct（Reason+Act）循环：think -> act(tool) -> observe -> 最多 N 步。

ReAct 输出格式约定：
    Thought: <推理>
    Action: <工具名>
    Action Input: <JSON 参数>

    或

    Thought: <推理>
    Final Answer: <最终答案>
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .context_manager import ContextManager
from .llm_interface import LLMInterface, create_llm
from .memory import LongTermMemory, ShortTermMemory, MemoryManager

# 模块级日志器：用于关键路径异常告警（替代静默 except pass）
_logger = logging.getLogger(__name__)


class Tool:
    """工具封装：名称、描述、可调用函数。"""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, **kwargs):
        return self.func(**kwargs)

    def to_schema(self) -> str:
        """返回工具说明（供 LLM 系统提示使用）。"""
        return f"- {self.name}: {self.description}"


class Workflow:
    """工作流封装：有序步骤序列。"""

    def __init__(self, name: str, description: str,
                 steps: List[Callable] = None):
        self.name = name
        self.description = description
        self.steps = steps or []

    def run(self, agent: "AerospaceAgent", **params) -> List[Any]:
        """依次执行各步骤，返回各步结果列表。"""
        results = []
        total = len(self.steps)
        for i, step in enumerate(self.steps, 1):
            print(f"  [工作流 {self.name}] 步骤 {i}/{total} ...")
            res = step(agent, **params)
            results.append(res)
        return results


# ----------------------------------------------------------------------
# 内置默认工具
# ----------------------------------------------------------------------
def _orbit_calculator(mission: str = "lunar_transfer",
                      altitude_km: float = 300) -> str:
    """计算轨道转移基础参数（二体近似）。"""
    MU_EARTH = 398600.4418  # km^3/s^2
    R_EARTH = 6378.137  # km
    r_moon = 384400.0  # km
    r_park = R_EARTH + altitude_km
    v_park = math.sqrt(MU_EARTH / r_park)
    a_t = (r_park + r_moon) / 2.0
    v_perigee = math.sqrt(MU_EARTH * (2.0 / r_park - 1.0 / a_t))
    dv_tli = v_perigee - v_park
    tof_hours = math.pi * math.sqrt(a_t ** 3 / MU_EARTH) / 3600.0
    return (
        f"mission={mission}, altitude={altitude_km}km | "
        f"停泊轨道速度={v_park:.3f} km/s, "
        f"TLI速度增量={dv_tli:.3f} km/s, "
        f"转移时间={tof_hours:.1f} h ({tof_hours / 24:.2f} d)"
    )


def _orbital_velocity(altitude_km: float = 400) -> str:
    """计算圆轨道速度与周期。"""
    MU_EARTH = 398600.4418
    R_EARTH = 6378.137
    r = R_EARTH + altitude_km
    v = math.sqrt(MU_EARTH / r)
    period = 2 * math.pi * r / v / 3600.0
    return f"高度 {altitude_km} km 圆轨道速度 = {v:.4f} km/s, 周期 = {period:.2f} h"


def _calculator(expression: str) -> str:
    """安全表达式计算器（允许数字、数学运算符和基本函数）。"""
    # 支持的字符：数字、运算符、括号、逗号、空格、字母（用于 math 函数和常量）
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,=<>!&|%~^ ]+", expression):
        return "错误：表达式包含非法字符"
    try:
        # 替换 ^ 为 **（幂运算）
        safe_expr = expression.replace("^", "**")
        # 将 math 函数注入命名空间
        value = eval(safe_expr, {"__builtins__": {}}, {"math": math, "pi": math.pi, "e": math.e})
        return f"{expression} = {value}"
    except Exception as e:
        return f"计算错误: {e}"


def default_tools() -> List[Tool]:
    """内置默认工具集。"""
    return [
        Tool("orbit_calculator", "计算轨道转移基础参数(mission, altitude_km)",
             _orbit_calculator),
        Tool("orbital_velocity", "计算圆轨道速度(altitude_km)", _orbital_velocity),
        Tool("calculator", "数学表达式计算器(expression)", _calculator),
    ]


# ----------------------------------------------------------------------
# 内置默认工作流
# ----------------------------------------------------------------------
def _wf_lunar_transfer(agent: "AerospaceAgent", **params) -> Dict[str, Any]:
    """地月转移轨道设计工作流：生成设计检查清单。"""
    print("  -> 生成地月转移轨道设计检查清单 ...")
    checklist = [
        "1. 发射窗口分析（月球黄经位置匹配）",
        "2. 停泊轨道参数确定（高度、倾角）",
        "3. 地月转移能量需求 C3 计算",
        "4. TLI 机动设计（速度增量大小与方向）",
        "5. 中途修正机动（TCM）规划，通常 2~3 次",
        "6. LOI 机动设计（捕获至月球轨道）",
    ]
    return {"checklist": checklist}


def _wf_orbit_design(agent: "AerospaceAgent", **params) -> Dict[str, Any]:
    """圆轨道参数设计工作流：计算速度与周期。"""
    alt = params.get("altitude_km", 400)
    res = agent.tools["orbital_velocity"](altitude_km=alt)
    print(f"  -> {res}")
    return {"velocity": res}


def default_workflows() -> List[Workflow]:
    """内置默认工作流集。"""
    return [
        Workflow("lunar_transfer", "地月转移轨道设计", [_wf_lunar_transfer]),
        Workflow("orbit_design", "圆轨道参数设计", [_wf_orbit_design]),
    ]


# ----------------------------------------------------------------------
# 简易 RAG（基于长期记忆）
# ----------------------------------------------------------------------
class SimpleRAG:
    """简易 RAG：基于 ``LongTermMemory`` 的文档检索。

    - ``index(dir)``  : 读取目录下文本文件，按段落切分后写入记忆
    - ``query(query)``: 检索相关段落
    """

    def __init__(self, memory: LongTermMemory = None):
        self.memory = memory or LongTermMemory()

    def index(self, dir_path: str) -> int:
        """索引目录下所有文本文件，返回入库段落数。"""
        path = Path(dir_path)
        if not path.exists():
            print(f"  目录不存在: {dir_path}")
            return 0
        count = 0
        for fp in path.rglob("*"):
            if fp.suffix.lower() not in (".txt", ".md", ".json", ".log"):
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, para in enumerate(text.split("\n\n")):
                para = para.strip()
                if len(para) < 5:
                    continue
                key = f"{fp.name}#{i}"
                self.memory.remember(key, para, tags=[str(fp)])
                count += 1
        self.memory.save()
        return count

    def query(self, query: str, top_k: int = 3) -> List[str]:
        """检索与 query 最相关的 top_k 个段落。"""
        results = self.memory.recall(query, top_k=top_k)
        return [f"[{k}] (相似度 {s:.3f})\n{v}" for k, s, v in results]


# ----------------------------------------------------------------------
# 主编排器
# ----------------------------------------------------------------------
class AerospaceAgent:
    """航天导航控制 Agent 主编排器。

    组合 LLM、上下文管理器、记忆、工具、工作流，通过 ReAct 循环完成任务。
    """

    def __init__(self, llm: LLMInterface = None,
                 context_manager: ContextManager = None,
                 memory: LongTermMemory = None,
                 tools: List[Tool] = None,
                 workflows: List[Workflow] = None,
                 max_steps: int = 10):
        self.llm = llm or create_llm()
        self.context_manager = context_manager or ContextManager()
        self.memory = memory or LongTermMemory()
        self.short_memory = ShortTermMemory()
        # MemoryManager 统一管理三层记忆 (可选,由 create_default_agent 装配)
        self.memory_manager: Optional[MemoryManager] = None
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.workflows: Dict[str, Workflow] = {w.name: w for w in (workflows or [])}
        self.max_steps = max_steps
        # MCP 工具注册表（BaseTool 实例，接口与原生 Tool 不同，单独存放）
        self.mcp_tools: Dict[str, Any] = {}
        # BaseWorkflow 实例（workflows/ 包中的预定义工作流，支持 execute() 直接执行）
        self.base_workflows: Dict[str, Any] = {}
        # 工具向量索引（按任务语义检索 top-K 工具，替代全量注入）
        self.tool_index: Optional[Any] = None
        # 可选挂载的 RAG（由 create_default_agent 设置）
        self.rag: Optional[SimpleRAG] = None
        # K5-H1: 初始化 system_prompt 属性，防止非工厂构造时 AttributeError
        self.system_prompt: Optional[str] = None
        self._basic_langchain_agent: Optional[Any] = None
        # 上下文窗口管理：token 预算与工具结果截断
        # max_context_tokens: 模型上下文窗口大小（默认 8192，适配大多数本地 vLLM 部署）
        #   可通过环境变量 AEROSPACE_MAX_CONTEXT_TOKENS 覆盖
        # max_output_tokens: LLM 单次输出最大 token 数（默认 1024，给 prompt 留空间）
        #   可通过环境变量 AEROSPACE_MAX_OUTPUT_TOKENS 覆盖
        # tool_result_max_chars: 单条工具结果最大字符数，超出则首尾截断
        self.max_context_tokens: int = int(
            os.environ.get("AEROSPACE_MAX_CONTEXT_TOKENS", 8192)
        )
        self.max_output_tokens: int = int(
            os.environ.get("AEROSPACE_MAX_OUTPUT_TOKENS", 1024)
        )
        self.tool_result_max_chars: int = 4000

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register_tool(self, tool: Tool) -> None:
        """注册单个原生工具。"""
        self.tools[tool.name] = tool

    def register_tools(self, tools: List[Tool]) -> None:
        """批量注册原生工具。"""
        for t in tools:
            self.register_tool(t)

    def register_mcp_tool(self, base_tool: Any) -> None:
        """注册一个 MCP 风格工具（BaseTool 实例，含可用性检测）。"""
        if getattr(base_tool, "name", None):
            self.mcp_tools[base_tool.name] = base_tool

    def register_workflow(self, workflow: Workflow) -> None:
        """注册单个工作流。"""
        self.workflows[workflow.name] = workflow

    def register_workflows(self, workflows: List[Workflow]) -> None:
        """批量注册工作流。"""
        for w in workflows:
            self.register_workflow(w)

    def register_base_workflow(self, bw: Any) -> None:
        """注册一个 BaseWorkflow 实例（来自 workflows/ 包，支持 execute() 直接执行）。"""
        if getattr(bw, "name", None):
            self.base_workflows[bw.name] = bw

    def register_base_workflows(self, bws: List[Any]) -> None:
        """批量注册 BaseWorkflow 实例。"""
        for bw in bws:
            self.register_base_workflow(bw)

    # ------------------------------------------------------------------
    # ReAct 解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_action(text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 输出中解析 Action / Action Input。"""
        action_blocks = list(
            re.finditer(
                r"Action:\s*(?P<tool>[^\r\n]+)(?P<body>.*?)(?=(?:\n\s*Action:\s*)|\Z)",
                text,
                re.DOTALL,
            )
        )
        if not action_blocks:
            return None

        block = action_blocks[-1]
        tool_name = block.group("tool").strip()
        input_match = re.search(r"Action Input:\s*(.+)", block.group("body"), re.DOTALL)
        raw_input = input_match.group(1).strip() if input_match else "{}"
        args = AerospaceAgent._parse_action_input(raw_input)
        return {"tool": tool_name, "args": args}

    @staticmethod
    def _parse_final_answer(text: str) -> Optional[str]:
        """从 LLM 输出中解析 Final Answer。"""
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    def _build_react_prompt(self, task: str) -> List[Dict[str, str]]:
        """构建 ReAct 系统提示与任务消息。"""
        tool_lines = [t.to_schema() for t in self.tools.values()]
        # 将 MCP 工具一并列出（带可用性标注：真实库/回退模式均可用）
        for name, bt in self.mcp_tools.items():
            src = getattr(bt, "source", "fallback")
            avail = "可用(真实)" if src == "real" else "可用(回退)"
            desc = getattr(bt, "description", "")
            tool_lines.append(f"- {name}: {desc} [{avail}]")
        tool_list = "\n".join(tool_lines) if tool_lines else "- (无可用工具)"
        context = self.context_manager.build_context(token_budget=8000)
        system = (
            "你是一个航天导航控制 Agent，使用 ReAct（Reason+Act）模式解决任务。\n\n"
            f"可用工具：\n{tool_list}\n\n"
            f"上下文：\n{context}\n\n"
            "回复格式（每次只输出一个 Thought，随后跟一个 Action 或 Final Answer）：\n"
            "Thought: <你的推理>\n"
            "Action: <工具名>\n"
            "Action Input: <JSON 参数>\n\n"
            "或者：\n"
            "Thought: <你的推理>\n"
            "Final Answer: <最终答案>\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def _execute_tool(self, tool_name: str, args: Any) -> str:
        """执行指定工具（原生 Tool 或 MCP BaseTool），返回结果字符串。

        统一入口：原生工具直接调用，MCP 工具走智能参数适配。
        错误返回使用 [TOOL ERROR] 前缀，便于 LLM 识别并换策略。
        """
        available = list(self.tools.keys()) + list(self.mcp_tools.keys())

        # 原生工具
        tool = self.tools.get(tool_name)
        if tool is not None:
            try:
                result = tool(**args) if isinstance(args, dict) else tool(input=args)
                return str(result)
            except TypeError as e:
                return (
                    f"[TOOL ERROR] 工具 '{tool_name}' 参数错误: {e}\n"
                    f"提示: 请检查参数名和类型，或调用 list_tools 查看 '{tool_name}' 的正确参数。"
                )
            except Exception as e:
                return (
                    f"[TOOL ERROR] 工具 '{tool_name}' 执行异常: {e}\n"
                    f"提示: 如再次失败，请换用其他工具或调用 list_tools。"
                )
        # MCP 工具（BaseTool.call(method, **kwargs) 接口）
        bt = self.mcp_tools.get(tool_name)
        if bt is not None:
            if not isinstance(args, dict):
                args = {"input": args}
            # 智能选择方法 + 参数适配
            method, adapted_kw, hint = self._adapt_mcp_call(bt, args)
            if hint:
                return (
                    f"[TOOL ERROR] 工具 '{tool_name}' 参数适配失败: {hint}\n"
                    f"提示: 请调用 list_tools 查看 '{tool_name}' 的正确用法。"
                )
            try:
                res = bt.call(method, **adapted_kw)
                return json.dumps(res, ensure_ascii=False, default=str)
            except TypeError as e:
                schema = self._get_method_schema(bt, method)
                return (
                    f"[TOOL ERROR] 工具 '{tool_name}' 参数错误: {e}\n"
                    f"  方法 '{method}' 期望参数: {schema}\n"
                    f"  实际传入: {list(adapted_kw.keys())}\n"
                    f"提示: 请调整参数后重试，或调用 list_tools 查看正确用法。"
                )
            except Exception as e:
                return (
                    f"[TOOL ERROR] 工具 '{tool_name}' 执行异常: {e}\n"
                    f"提示: 如再次失败，请换用其他工具。"
                )
        # 未知工具
        avail_preview = ", ".join(available[:20])
        if len(available) > 20:
            avail_preview += f" ...等共 {len(available)} 个工具"
        return (
            f"[TOOL ERROR] 工具 '{tool_name}' 不存在。\n"
            f"可用工具: {avail_preview}\n"
            f"提示: 请调用 list_tools 查看完整工具列表和参数说明。"
        )

    # ------------------------------------------------------------------
    # 上下文窗口管理：token 预算强制执行
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_messages_tokens(messages: List[Dict]) -> int:
        """估算消息列表的总 token 数（保守估计，×4/3）。

        使用 rough_token_count（字符数 / 3.5），与 CCB token_estimation 对齐。
        """
        from .token_estimation import rough_token_count
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += rough_token_count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += rough_token_count(block.get("text", "") or block.get("content", "") or "")
                    else:
                        total += rough_token_count(str(block))
            else:
                total += rough_token_count(str(content))
        # 保守系数 4/3（与 CCB 一致）
        return int(total * 4.0 / 3.0)

    def _enforce_token_budget(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
    ) -> List[Dict]:
        """强制执行 token 预算——超限时渐进式压缩消息列表。

        三级策略（按破坏性从低到高）：
          Phase 1: 清除旧 Observation 中的工具结果内容（保留最近 3 条）
          Phase 2: 截断过长的单条消息（超过 ~2000 token 的）
          Phase 3: 丢弃最旧的非系统消息（保留最近 6 条）

        硬性保证：system 消息永不丢弃（包含任务规格、Essential 层）。
        """
        max_tokens = max_tokens or self.max_context_tokens
        # safe_budget = 总窗口 - LLM 输出预留，确保 prompt + completion <= 窗口
        safe_budget = max(512, max_tokens - self.max_output_tokens)

        current = self._estimate_messages_tokens(messages)
        if current <= safe_budget:
            return messages

        # 分离 system 和非 system 消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Phase 1: 清除旧 Observation / tool 结果内容（保留最近 2 条）
        obs_indices = [
            i for i, m in enumerate(non_system)
            if "Observation:" in m.get("content", "") or m.get("role") == "tool"
        ]
        if len(obs_indices) > 2:
            for i in obs_indices[:-2]:
                non_system[i] = {
                    **non_system[i],
                    "content": "[旧工具结果已清除以节省上下文空间]",
                }

        current = self._estimate_messages_tokens(system_msgs + non_system)
        if current <= safe_budget:
            return system_msgs + non_system

        # Phase 2: 截断过长的单条消息（> 7000 字符 ≈ 2000 token）
        for i, m in enumerate(non_system):
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > 7000:
                non_system[i] = {
                    **m,
                    "content": content[:7000]
                    + f"\n...[已截断，原长度 {len(content)} 字符]...",
                }

        current = self._estimate_messages_tokens(system_msgs + non_system)
        if current <= safe_budget:
            return system_msgs + non_system

        # Phase 3: 丢弃最旧的非系统消息（保留最近 6 条 = 3 轮对话）
        min_keep = 6
        while (
            len(non_system) > min_keep
            and self._estimate_messages_tokens(system_msgs + non_system) > safe_budget
        ):
            non_system.pop(0)

        # Phase 4: 仍然超标——截断剩余消息中最大的 Observation 到 500 字符
        current = self._estimate_messages_tokens(system_msgs + non_system)
        if current > safe_budget:
            for i, m in enumerate(non_system):
                content = m.get("content", "")
                if isinstance(content, str) and "Observation:" in content and len(content) > 500:
                    non_system[i] = {
                        **m,
                        "content": content[:500] + "...[紧急截断]",
                    }

        # 如果删除后第一条是 assistant，插入 user 占位符保持角色交替
        if non_system and non_system[0].get("role") == "assistant":
            non_system.insert(0, {"role": "user", "content": "[更早的对话已压缩]"})

        return system_msgs + non_system

    @staticmethod
    def _truncate_tool_result(result: Any, max_chars: int = 4000) -> str:
        """截断过长的工具结果，保留首尾摘要。

        策略：保留前 60% 和后 20%，中间用省略标记。
        """
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                result = str(result)
        if len(result) <= max_chars:
            return result
        head_size = int(max_chars * 0.6)
        tail_size = int(max_chars * 0.2)
        return (
            result[:head_size]
            + f"\n...[已截断 {len(result) - head_size - tail_size} 字符]...\n"
            + result[-tail_size:]
        )

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        """判断异常是否为上下文长度超限导致的请求失败。"""
        msg = str(exc).lower()
        return any(kw in msg for kw in (
            "400", "bad request", "context length", "too long",
            "maximum context", "token limit", "prompt is too long",
            "request too large", "payload too large",
            "exceeds the available context size", "exceeds context",
            "context size", "max tokens", "exceeds",
        ))

    # ------------------------------------------------------------------
    # 智能参数适配 (从 CEOEngine 迁移,统一工具执行入口)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_action_input(raw: str) -> dict:
        """智能解析 Action Input — 多策略。

        策略链：JSON → ast.literal_eval → 安全 eval → 回退。
        """
        raw = raw.strip()
        decoder = json.JSONDecoder()
        try:
            result, _ = decoder.raw_decode(raw)
            if isinstance(result, dict):
                return result
            return {"input": result}
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
            return {"input": result}
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, dict):
                return result
            return {"input": result}
        except (ValueError, SyntaxError):
            pass
        try:
            safe_ns = {
                "math": math, "pi": math.pi, "e": math.e,
                "abs": abs, "sqrt": math.sqrt, "pow": pow,
                "min": min, "max": max, "round": round,
                "__builtins__": {},
            }
            result = eval(raw, safe_ns, {})  # noqa: S307
            if isinstance(result, dict):
                return result
            return {"input": result}
        except Exception:
            pass
        return {"input": raw}

    @staticmethod
    def _action_signature(tool_name: str, args: Any) -> str:
        """Build a stable signature for repeated-action detection."""
        try:
            args_text = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            args_text = str(args)
        return f"{tool_name}:{args_text}"

    @staticmethod
    def _get_method_schema(bt, method: str) -> str:
        """获取工具方法的参数 schema。"""
        schema = getattr(bt, "methods_schema", {})
        if method in schema:
            params = schema[method].get("params", {})
            return ", ".join(f"{k}: {v}" for k, v in params.items())
        return "(未知)"

    def _adapt_mcp_call(self, bt, args: dict) -> Tuple[str, dict, str]:
        """智能适配 MCP 工具调用。返回 (method, adapted_kwargs, error_hint)。"""
        method = args.pop("method", None)
        if not method:
            method = self._select_best_method(bt, args)
        if not method:
            methods = getattr(bt, "list_methods", lambda: [])()
            return "", args, f"工具 '{bt.name}' 无可用方法。方法列表: {methods}"
        adapted = self._map_params(bt, method, args)
        return method, adapted, ""

    def _select_best_method(self, bt, args: dict) -> str:
        """根据传入参数选择最佳匹配方法。"""
        schema = getattr(bt, "methods_schema", {})
        if not schema:
            methods = getattr(bt, "list_methods", lambda: [])()
            return methods[0] if methods else ""
        arg_keys = set(args.keys())
        best_method = ""
        best_score = -1
        for mname, mschema in schema.items():
            param_keys = set(mschema.get("params", {}).keys())
            overlap = len(arg_keys & param_keys)
            extra = len(arg_keys - param_keys)
            score = overlap * 2 - extra
            if score > best_score:
                best_score = score
                best_method = mname
        if best_score <= 0:
            methods = list(schema.keys())
            return methods[0] if methods else ""
        return best_method

    def _map_params(self, bt, method: str, args: dict) -> dict:
        """参数名映射 — 将 LLM 友好参数名映射到工具期望参数名。"""
        schema = getattr(bt, "methods_schema", {})
        mschema = schema.get(method, {})
        expected_params = mschema.get("params", {})
        arg_keys = set(args.keys())
        expected_keys = set(expected_params.keys())
        if arg_keys and arg_keys.issubset(expected_keys):
            return dict(args)
        if method == "simulate" and bt.name == "basilisk":
            return self._adapt_basilisk_simulate(args, expected_params)
        if "scenario_config" in expected_keys and "scenario_config" not in args:
            config = {}
            duration = None
            for k, v in args.items():
                if k in ("duration_s", "duration", "sim_duration"):
                    duration = v
                elif k in expected_keys:
                    pass
                else:
                    config[k] = v
            result = {}
            if config:
                result["scenario_config"] = config
            if duration is not None and "duration" in expected_keys:
                result["duration"] = self._to_float(duration)
            for k in arg_keys & expected_keys:
                result[k] = args[k]
            if result:
                return result
        if len(expected_keys) == 1:
            only_param = list(expected_keys)[0]
            if only_param not in args:
                return {only_param: dict(args)}
        return dict(args)

    @staticmethod
    def _adapt_basilisk_simulate(args: dict, expected_params: dict) -> dict:
        """BasiliskTool.simulate 专用参数适配。"""
        config = {}
        duration = None
        for k, v in args.items():
            if k in ("duration_s", "duration", "sim_duration"):
                duration = v
            elif k in ("initial_state", "state", "state_vec"):
                if isinstance(v, dict):
                    r = v.get("r", v.get("position", [0, 0, 0]))
                    vel = v.get("v", v.get("velocity", [0, 0, 0]))
                    state = list(r) + list(vel)
                elif isinstance(v, (list, tuple)):
                    state = list(v)
                else:
                    state = [0, 0, 0, 0, 0, 0]
                config["spacecraft"] = [{"state": state, "dynamics": "point_mass"}]
            elif k in ("force_model", "forces"):
                fm = v if isinstance(v, dict) else {"model": str(v)}
                if "mu" in fm:
                    config["mu"] = fm["mu"]
                config["force_model"] = fm
            elif k in ("dt", "step_size", "output_step_s"):
                config["dt"] = v
            else:
                config[k] = v
        result = {}
        if config:
            result["scenario_config"] = config
        if duration is not None and "duration" in expected_params:
            result["duration"] = float(duration)
        return result

    @staticmethod
    def _to_float(val) -> float:
        """安全转 float。"""
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # 工作流匹配
    # ------------------------------------------------------------------
    def _match_workflow(self, task: str) -> Optional[Workflow]:
        """根据任务文本简单匹配工作流。"""
        low = task.lower()
        # 先按名称/描述精确匹配
        for wf in self.workflows.values():
            if wf.name and (wf.name in task or wf.name.lower() in low):
                return wf
        # 再按领域关键词匹配
        if "地月" in task or "月球" in task or "lunar" in low:
            return self.workflows.get("lunar_transfer")
        if "轨道" in task or "orbit" in low:
            return self.workflows.get("orbit_design")
        return None

    # ------------------------------------------------------------------
    # 主运行循环
    # ------------------------------------------------------------------
    def run(self, task: str) -> str:
        """执行任务：自动调用工作流与工具，运行 ReAct 循环。

        Args:
            task: 用户任务描述

        Returns:
            最终答案文本
        """
        print("\n========== 任务开始 ==========")
        print(f"任务: {task}")

        # Essential 层：任务规格原样保留（硬性要求，永不压缩）
        self.context_manager.add_essential(f"【用户原始任务】{task}")
        self.context_manager.add_essential(
            "【设计约束】任务规格与关键参数不得在后续上下文压缩中丢失或失真。"
        )

        # 1. 尝试匹配并运行工作流（结果存入上下文与记忆）
        matched_wf = self._match_workflow(task)
        if matched_wf:
            print(f"[匹配工作流] {matched_wf.name}: {matched_wf.description}")
            wf_results = matched_wf.run(self)
            for r in wf_results:
                self.context_manager.add_tool_record(
                    f"workflow:{matched_wf.name}", {}, r)

        # 2. ReAct 循环
        self.short_memory.add("user", task)
        messages = self._build_react_prompt(task)
        observation = ""

        for step in range(1, self.max_steps + 1):
            print(f"\n--- ReAct 第 {step}/{self.max_steps} 步 ---")
            # K5-H2: 累积历史消息（与 run_react_stream 对齐），而非每步重建
            if observation:
                messages.append({"role": "user", "content": observation})

            # Token 预算强制执行——防止上下文窗口溢出
            messages = self._enforce_token_budget(messages)

            try:
                response = self.llm.chat(messages)
            except Exception as e:
                # 上下文超长：压缩后重试一次
                if self._is_context_length_error(e):
                    messages = self._enforce_token_budget(
                        messages, max_tokens=int(self.max_context_tokens * 0.6)
                    )
                    try:
                        response = self.llm.chat(messages)
                    except Exception as e2:
                        print(f"[LLM 调用失败(重试后)] {e2}")
                        return f"任务失败：LLM 调用异常 - {e2}"
                else:
                    print(f"[LLM 调用失败] {e}")
                    return f"任务失败：LLM 调用异常 - {e}"

            # 打印思考（截断过长输出）
            preview = response if len(response) <= 500 else response[:500] + " ..."
            print(f"[Thought]\n{preview}")
            self.short_memory.add("assistant", response)
            self.context_manager.add_message("assistant", response)
            # K5-H2: 将 LLM 回复累积到 messages，保持多步推理上下文
            messages.append({"role": "assistant", "content": response})

            # 解析最终答案
            final = self._parse_final_answer(response)
            if final:
                print(f"\n[最终答案]\n{final}")
                self.context_manager.add_message("assistant", f"FINAL: {final}")
                self._persist_memory(task, final)
                print("========== 任务完成 ==========\n")
                return final

            # 解析动作
            action = self._parse_action(response)
            if not action:
                # 既无 Action 也无 Final Answer：以回复作为最终答案
                print("\n[未检测到 Action/Final Answer，以回复作为最终答案]")
                self._persist_memory(task, response)
                print("========== 任务完成 ==========\n")
                return response

            tool_name = action["tool"]
            args = action["args"]
            print(f"[Action] 调用工具: {tool_name}, 参数: {args}")
            result = self._execute_tool(tool_name, args)
            result = self._truncate_tool_result(result, self.tool_result_max_chars)
            print(f"[Observation] {result}")
            self.context_manager.add_tool_record(tool_name, args, result)
            self.short_memory.add("tool", result)
            observation = f"Observation: {result}"

        print(f"\n[达到最大步数 {self.max_steps}，终止循环]")
        print("========== 任务结束 ==========\n")
        return "已达到最大推理步数，任务未得出最终答案。"

    # ------------------------------------------------------------------
    # 快速执行路径 — 优先直接运行 BaseWorkflow，回退到精简 ReAct
    # ------------------------------------------------------------------
    # BaseWorkflow 关键词匹配表（顺序即优先级：特定任务在前，通用在后）
    _BW_KEYWORD_MAP: Dict[str, List[str]] = {
        "lunar_transfer": ["地月转移", "月球转移", "lunar transfer",
                           "月球轨道", "登月", "tle"],
        "launch_window": ["发射窗口", "launch window", "窗口分析"],
        "basilisk_viz": ["可视化", "仿真", "visualization", "basilisk", "3d"],
        "literature_review": ["文献", "论文", "literature", "paper", "综述"],
        "orbit_design": ["轨道设计", "轨道参数", "orbit design", "leo", "geo",
                         "sso", "molniya", "圆轨道", "静止轨道", "太阳同步"],
    }

    def _match_base_workflow(self, task: str) -> Optional[str]:
        """根据任务文本匹配 BaseWorkflow 名称（优先级高者优先）。"""
        low = task.lower()
        best_name: Optional[str] = None
        best_score = 0
        for wf_name, keywords in self._BW_KEYWORD_MAP.items():
            if wf_name not in self.base_workflows:
                continue
            score = sum(1 for kw in keywords if kw in task or kw in low)
            # 使用 >= 让后出现的同名分也不覆盖先出现的（保持表顺序优先级）
            if score > best_score:
                best_score = score
                best_name = wf_name
        return best_name if best_score > 0 else None

    def run_fast(self, task: str, **wf_params) -> str:
        """快速执行路径——优先直接运行匹配的 BaseWorkflow。

        流程：
          1. 匹配预定义 BaseWorkflow → 直接 execute()（0 次 LLM 调用，最快）
          2. 匹配失败 → 回退到精简 ReAct 循环 run_react_fast()

        Args:
            task: 用户任务描述
            **wf_params: 传递给 BaseWorkflow.execute() 的参数

        Returns:
            结果文本
        """
        print(f"\n========== 快速执行 ==========")
        print(f"任务: {task}")

        # 1. 尝试匹配 BaseWorkflow
        wf_name = self._match_base_workflow(task)
        if wf_name:
            bw = self.base_workflows[wf_name]
            print(f"[匹配工作流] {wf_name}: {getattr(bw, 'description', '')}")
            try:
                result = bw.execute(**wf_params)
                return self._format_workflow_result(wf_name, result)
            except Exception as e:
                print(f"[工作流执行失败: {e}] 回退到 ReAct")

        # 2. 回退到精简 ReAct
        print("[未匹配工作流] 使用精简 ReAct 循环")
        return self.run_react_fast(task)

    @staticmethod
    def _format_workflow_result(wf_name: str, result: Any) -> str:
        """将 WorkflowResult 格式化为可读文本。"""
        if hasattr(result, "success"):
            lines = [
                f"## 工作流: {wf_name}",
                f"状态: {'成功' if result.success else '失败'}",
                f"摘要: {result.summary}",
            ]
            if result.steps_log:
                lines.append("\n### 执行步骤:")
                for s in result.steps_log:
                    icon = "OK" if s.get("status") == "success" else "FAIL"
                    lines.append(f"  [{icon}] {s.get('step','')}: {s.get('detail','')}")
            if result.artifacts:
                lines.append("\n### 产出文件:")
                for a in result.artifacts:
                    lines.append(f"  - {a}")
            if result.result and isinstance(result.result, dict):
                lines.append("\n### 关键结果:")
                for k, v in list(result.result.items())[:10]:
                    lines.append(f"  - {k}: {v}")
            return "\n".join(lines)
        return str(result)

    # ------------------------------------------------------------------
    # LangChain Agent 循环 —— 基于 langchain-core 的标准化 ReAct
    # ------------------------------------------------------------------
    def run_langchain(
        self,
        task: str,
        max_steps: int = 20,
        stream_callback: Optional[Callable[[str], None]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Run the minimal LangChain-oriented agent entrypoint.

        The old implementation tried to rebuild a full ReAct executor on top of
        langchain-core. That path is now legacy for the basic framework stage:
        basic tasks must not enter a recursive tool loop. This method keeps the
        public name stable while delegating to BasicLangChainAgent.
        """
        try:
            from ..langchain_agent import BasicAgentConfig, create_basic_langchain_agent
        except Exception as exc:
            return f"LangChain 基础入口加载失败: {exc}"

        if self._basic_langchain_agent is None:
            config = BasicAgentConfig(max_output_tokens=self.max_output_tokens)
            self._basic_langchain_agent = create_basic_langchain_agent(
                llm=self.llm,
                workspace=Path.cwd(),
                config=config,
                rag=getattr(self, "rag", None),
                mcp_tools=getattr(self, "mcp_tools", None),
                skill_registry=getattr(self, "skills", None),
                skill_agent=self,
            )
        else:
            self._basic_langchain_agent.set_interfaces(
                rag=getattr(self, "rag", None),
                mcp_tools=getattr(self, "mcp_tools", None),
                skill_registry=getattr(self, "skills", None),
                skill_agent=self,
            )
        agent = self._basic_langchain_agent
        result = agent.invoke(task)
        text = result.to_text()

        if stream_callback:
            try:
                stream_callback(text)
            except Exception:
                pass

        if result.ok:
            try:
                self._persist_memory(task, text)
            except Exception:
                pass

        return text

    def run_langchain_react(
        self,
        task: str,
        max_steps: int = 20,
        stream_callback: Optional[Callable[[str], None]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Backward-compatible alias for older callers."""
        return self.run_langchain(
            task=task,
            max_steps=max_steps,
            stream_callback=stream_callback,
            system_prompt=system_prompt,
        )

    def run_react_fast(self, task: str, max_steps: int = 6) -> str:
        """精简 ReAct 循环——去除重上下文管理/记忆/指标开销，专注快速推理。

        与 run_react_stream 的区别：
        - 无 Phase A 蓝图注入
        - 无 MemoryManager 召回/工作记忆
        - 无 metrics 观测性埋点
        - 无 CEO 三层上下文管理（仅 system + 对话）
        - 默认 6 步上限（非 9999）
        - system prompt 精简（只含工具列表 + 格式说明）

        Args:
            task: 用户任务描述
            max_steps: 最大推理步数（默认 6）

        Returns:
            最终答案文本
        """
        print(f"\n--- 精简 ReAct (max={max_steps}) ---")
        self.short_memory.add("user", task)

        # 工具发现：向量检索 top-K 相关工具（替代全量 105 个注入）
        # 始终保留 create_tool（自进化入口）
        always_include = {"create_tool", "list_tools", "tool_help"}

        if self.tool_index and self.tool_index.is_built:
            # 语义检索 top-K
            from ..research_tools import get_registry as _get_rt
            _rt = _get_rt()
            hits = self.tool_index.search(task, k=8)
            tool_names = set(h["name"] for h in hits) | always_include
            schemas = []
            for name in sorted(tool_names):
                tool = _rt.get(name)
                if tool:
                    schemas.append(tool.to_schema())
                elif name in self.tools:
                    schemas.append(self.tools[name].to_schema())
            tool_list = "\n".join(schemas) if schemas else "- (无可用工具)"
            tool_count = len(schemas)
        else:
            # 回退：全量按分类展示
            from collections import defaultdict
            _cat_tools: Dict[str, List[str]] = defaultdict(list)
            for name, t in self.tools.items():
                _cat_tools["general"].append(t.to_schema())
            try:
                from ..research_tools import get_registry as _get_rt
                _rt = _get_rt()
                for cat, tools in _rt.categories().items():
                    _cat_tools[cat] = [f"- {t}" for t in tools]
            except Exception:
                pass
            tool_sections = []
            for cat, lines in sorted(_cat_tools.items()):
                if lines:
                    tool_sections.append(f"  [{cat}] ({len(lines)})\n    " + "\n    ".join(lines))
            tool_list = "\n".join(tool_sections) if tool_sections else "- (无可用工具)"
            tool_count = sum(len(v) for v in _cat_tools.values())

        system = (
            "你是航天动力学 Agent，使用 ReAct 模式解决任务。\n\n"
            f"可用工具（{tool_count} 个，按相关度精选）：\n{tool_list}\n\n"
            "回复格式：\n"
            "Thought: <推理>\n"
            "Action: <工具名>\n"
            "Action Input: <JSON 参数>\n\n"
            "或：\n"
            "Thought: <推理>\n"
            "Final Answer: <最终答案>\n\n"
            "提示：如果需要的工具不在列表中，调用 list_tools 查看更多，"
            "或调用 create_tool 创建新工具。\n"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        observation = ""
        for step in range(1, max_steps + 1):
            print(f"  [ReAct 第 {step}/{max_steps} 步]")
            if observation:
                messages.append({"role": "user", "content": observation})

            try:
                response = self.llm.chat(messages)
            except Exception as e:
                return f"任务失败：LLM 调用异常 - {e}"

            if not response.strip():
                return "LLM 返回空响应，终止推理。"

            messages.append({"role": "assistant", "content": response})

            # 解析 Final Answer
            final = self._parse_final_answer(response)
            if final:
                print(f"  [最终答案] {final[:200]}")
                self._persist_memory(task, final)
                return final

            # 解析 Action
            action = self._parse_action(response)
            if not action:
                self._persist_memory(task, response)
                return response

            tool_name = action["tool"]
            args = action["args"]
            if isinstance(args, str):
                args = self._parse_action_input(args)

            print(f"  [Action] {tool_name}, 参数: {args}")
            result = self._execute_tool(tool_name, args)
            result = self._truncate_tool_result(result, self.tool_result_max_chars)
            print(f"  [Observation] {str(result)[:200]}")
            observation = f"Observation: {result}"

            # 错误时注入强化提示——与 run_react_stream 对齐
            if "错误" in result or "error" in result.lower() or "[TOOL ERROR]" in result:
                observation += (
                    "\n[系统提示] 工具调用失败。"
                    "你必须换用其他工具或调整参数，不得重复调用同一错误工具。"
                )

        print(f"  [达到最大步数 {max_steps}]")
        return f"已达到最大推理步数({max_steps})，未得出最终答案。"

    def _persist_memory(self, task: str, answer: str) -> None:
        """将任务与答案持久化到长期记忆。"""
        try:
            self.memory.remember(task, answer, tags=["task_result"])
            self.memory.save()
        except Exception as e:
            print(f"[记忆持久化失败] {e}")

    # ------------------------------------------------------------------
    # 原生 Function Calling 循环（参考 Claude Code 架构）
    # ------------------------------------------------------------------
    def run_native(self, task: str, max_steps: int = 10) -> str:
        """原生 Function Calling Agent 循环。

        与 run_react_fast 的根本区别：
        - LLM 直接输出结构化 tool_call，不用正则解析文本
        - 工具结果以 tool role 消息返回（OpenAI 标准）
        - system prompt 不注入工具列表（工具通过 tools 参数传递）
        - 工具定义静态锁定（利用 prompt caching）

        参考: Anthropic Claude Code 的 agentic loop 架构
        """
        print(f"\n========== 原生 Function Calling ==========")
        print(f"任务: {task}")

        # 1. 构建工具定义（OpenAI 格式）
        tools_def = self._build_tools_for_native(task)
        if not tools_def:
            print("[无可用工具] 回退到 run_react_fast")
            return self.run_react_fast(task, max_steps=max_steps)

        print(f"  工具数: {len(tools_def)} (向量检索精选)")

        # 2. 构建 messages（静态 system + 动态 history）
        system = (
            "你是航天动力学科研 Agent。根据用户需求调用工具完成任务。\n"
            "规则：\n"
            "1. 先分析需要什么信息，调用相应工具\n"
            "2. 工具返回后判断是否需要更多操作\n"
            "3. 任务完成后给出最终答案\n"
            "4. 如果工具不存在，调用 create_tool 创建新工具\n"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        # 3. Agent 循环
        for step in range(1, max_steps + 1):
            print(f"\n  [Step {step}/{max_steps}]")

            try:
                resp = self.llm.chat_with_tools(messages, tools_def, timeout=120)
            except Exception as e:
                print(f"  [LLM 调用失败] {e}")
                return f"任务失败：LLM 调用异常 - {e}"

            content = resp.get("content")
            tool_calls = resp.get("tool_calls")
            finish_reason = resp.get("finish_reason", "stop")

            # 如果有文本内容，打印
            if content:
                print(f"  [思考] {content[:200]}")

            # 没有 tool_calls → 任务完成
            if not tool_calls:
                answer = content or "(无输出)"
                print(f"  [完成] finish_reason={finish_reason}")
                self._persist_memory(task, answer)
                return answer

            # 将 assistant 的 tool_calls 加入 messages
            assistant_msg = {"role": "assistant", "content": content}
            # 保留原始 tool_calls 格式供 API 使用
            raw_calls = []
            for tc in tool_calls:
                raw_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                    },
                })
            assistant_msg["tool_calls"] = raw_calls
            messages.append(assistant_msg)

            # 执行每个 tool_call
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                call_id = tc["id"]

                print(f"  [调用] {tool_name}({tool_args})")

                # 执行工具
                result = self._execute_tool(tool_name, tool_args)

                # 大输出写文件（参考 Claude Code 的 dynamic context discovery）
                result_str = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
                if len(result_str) > 2000:
                    import os as _os
                    import tempfile as _tf
                    _tmp = _os.path.join(_tf.gettempdir(), f"tool_out_{call_id}.txt")
                    with open(_tmp, "w", encoding="utf-8") as _f:
                        _f.write(result_str)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"输出已写入文件: {_tmp} (共 {len(result_str)} 字符)。摘要: {result_str[:500]}...",
                    }
                    print(f"  [结果] 输出过大({len(result_str)}字符)，写入 {_tmp}")
                else:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_str,
                    }
                    print(f"  [结果] {result_str[:200]}")

                messages.append(tool_msg)

        print(f"\n  [达到最大步数 {max_steps}]")
        return f"已达到最大步数({max_steps})，未得出最终答案。"

    def _build_tools_for_native(self, task: str) -> List[Dict]:
        """构建 OpenAI 格式工具定义——向量检索 top-K + 元工具。

        参考 Claude Code: 工具定义静态锁定，不放入 system prompt。
        """
        from ..research_tools import get_registry as _get_rt
        _rt = _get_rt()

        always_include = {"create_tool", "list_tools", "tool_help"}
        tool_names = set()

        # 向量检索 top-K
        if self.tool_index and self.tool_index.is_built:
            hits = self.tool_index.search(task, k=10)
            tool_names = set(h["name"] for h in hits)

        tool_names |= always_include

        # 构建 OpenAI 格式
        tools_def = []
        for name in sorted(tool_names):
            tool = _rt.get(name)
            if tool is None and name in self.tools:
                # 原生 Tool
                tools_def.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": self.tools[name].description,
                        "parameters": {"type": "object", "properties": {}},
                    },
                })
            elif tool:
                # ResearchTool → 转 JSON Schema
                props = {}
                required = []
                for p in tool.params:
                    props[p.name] = {
                        "type": p.type,
                        "description": p.description,
                    }
                    if p.default is not None:
                        props[p.name]["default"] = p.default
                    if p.required:
                        required.append(p.name)
                tools_def.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": required,
                        },
                    },
                })
        return tools_def

    # ------------------------------------------------------------------
    # QueryEngine 驱动（1:1 复刻 CCB 架构）
    # ------------------------------------------------------------------
    def _build_tool_interfaces(self) -> List[Any]:
        """构建 ToolInterface 列表 — 将所有工具适配为新接口。"""
        from .tool_adapter import (
            wrap_callable_tools,
            wrap_research_tools,
        )
        interfaces = []
        # 1. 原生 Tool (name, description, func)
        interfaces.extend(wrap_callable_tools(list(self.tools.values())))
        # 2. ResearchTool (dataclass)
        try:
            from ..research_tools import get_registry as _get_rt
            _rt = _get_rt()
            interfaces.extend(wrap_research_tools(_rt))
        except Exception:
            pass
        return interfaces

    def run_query_engine(
        self,
        task: str,
        max_turns: int = 25,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """QueryEngine 驱动的 Agent 循环 — 1:1 复刻 CCB query loop。

        替代 run_native()，使用 QueryEngine + query() 异步生成器架构。

        流程（照搬 CCB）：
        1. 构建 ToolInterface 列表 + OpenAI tools_def
        2. 创建 QueryEngineConfig
        3. QueryEngine.submit_message() 异步生成器
        4. 收集 SDKMessage → 提取最终结果

        Args:
            task: 用户任务
            max_turns: 最大轮数
            stream_callback: 流式输出回调

        Returns:
            最终结果文本
        """
        import asyncio

        # 1. 构建 ToolInterface 列表
        tool_interfaces = self._build_tool_interfaces()
        if not tool_interfaces:
            # 回退到 run_react_fast
            return self.run_react_fast(task, max_steps=max_turns)

        # 2. 向量检索 top-K 工具
        from .tool_adapter import build_tools_def_for_query
        tool_names = None
        if self.tool_index and self.tool_index.is_built:
            hits = self.tool_index.search(task, k=10)
            tool_names = set(h["name"] for h in hits)
            # 始终包含元工具
            tool_names |= {"create_tool", "list_tools", "tool_help"}

        tools_def = build_tools_def_for_query(tool_interfaces, tool_names)
        if not tools_def:
            return self.run_react_fast(task, max_steps=max_turns)

        # 3. 构建 QueryEngineConfig
        from .query_engine import QueryEngine, QueryEngineConfig
        config = QueryEngineConfig(
            tools=tool_interfaces,
            llm=self.llm,
            tools_def=tools_def,
            max_turns=max_turns,
            stream_callback=stream_callback,
            verbose=False,
        )

        # 4. 运行 QueryEngine
        engine = QueryEngine(config)

        result_text = ""
        errors = []

        async def _run():
            nonlocal result_text, errors
            async for msg in engine.submit_message(task):
                if msg.type == "assistant":
                    # 不在此处流式输出 — query.py 的伪流式已处理
                    # 仅记录最后一条助手消息用于结果提取
                    pass
                elif msg.type == "progress":
                    # 工具执行进度
                    data = msg.data
                    if data.get("type") == "activity":
                        desc = data.get("description", "")
                        if desc:
                            print(f"\n  [工具] {desc}", flush=True)
                elif msg.type == "user":
                    # 工具结果消息
                    pass
                elif msg.type == "result":
                    if msg.is_error:
                        errors.extend(msg.errors)
                    result_text = msg.result or ""
                    if not stream_callback and result_text:
                        print()  # 换行

        try:
            asyncio.run(_run())
        except RuntimeError:
            # 已经在事件循环中 — 使用 nest_asyncio 或直接运行
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 创建任务并等待
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _run)
                    future.result()
            else:
                loop.run_until_complete(_run())

        # 5. 持久化记忆
        if result_text and not errors:
            self._persist_memory(task, result_text)

        if errors and not result_text:
            return f"任务失败: {'; '.join(errors)}"

        return result_text

    # ------------------------------------------------------------------
    # 流式 ReAct 循环 (统一入口,支持蓝图注入 + 上下文管理 + 记忆)
    # ------------------------------------------------------------------
    def run_react_stream(
        self,
        task: str,
        blueprint: Optional[Dict] = None,
        max_steps: int = 20,
        stream_callback: Optional[Callable[[str], None]] = None,
        enable_context: bool = True,
    ) -> str:
        """流式 ReAct 循环 — 统一的推理执行入口。

        集成 CEO 三层上下文管理 + 短期/长期记忆 + 蓝图注入 + 智能工具执行。
        CEOEngine 和直接调用者均走此入口,消除双循环问题。

        Args:
            task: 用户任务描述
            blueprint: Phase A 产出的 v1 蓝图 (可选)
            max_steps: 最大推理步数 (默认 20,防止坏循环无限扩散)
            stream_callback: 流式输出回调 (chunk -> None)
            enable_context: 是否启用上下文管理 (测试可关闭)

        Returns:
            最终答案字符串
        """
        # --- 观测性埋点 (懒加载) ---
        from ..utils.observability import get_logger, get_metrics
        log = get_logger("agent")
        metrics = get_metrics()
        log.info("react_stream_start", data={"task": task[:200], "max_steps": max_steps})
        _steps_used = 0

        # --- 初始化任务上下文 ---
        if enable_context:
            self.context_manager.add_essential(f"任务: {task}")
        self.short_memory.add("user", task)

        # MemoryManager 任务生命周期: 启动任务 + 召回相关长期记忆
        if self.memory_manager is not None:
            self.memory_manager.start_task(task)
            # 召回与任务相关的长期记忆,注入工作记忆
            try:
                recalled = self.memory_manager.recall_to_working(task, top_k=3)
                if recalled:
                    self.context_manager.add_essential(
                        f"相关记忆: {recalled}"
                    )
            except Exception:
                pass

        blueprint_text = ""
        if blueprint:
            bp = blueprint.get("blueprint", blueprint)
            blueprint_text = self._format_blueprint(bp)

        # --- 构建 system prompt ---
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        if blueprint_text:
            system_parts.append(f"\n## Phase A 蓝图\n{blueprint_text}")
        if enable_context:
            ctx = self.context_manager.build_context(token_budget=8000)
            if ctx:
                system_parts.append(f"\n## 上下文\n{ctx}")
        # K2.4: 技能接入 ReAct —— 将技能描述注入 system prompt
        if hasattr(self, "skills") and self.skills and hasattr(self.skills, "list_skills"):
            try:
                skill_list = self.skills.list_skills()
                if skill_list:
                    skill_desc = "\n".join(
                        f"  - **{s['name']}**: {s.get('description', '')}"
                        for s in skill_list)
                    system_parts.append(
                        f"\n## 可用技能\n以下技能可通过调用相应工具触发:\n{skill_desc}")
            except Exception:
                pass
        system_prompt = "\n".join(system_parts)

        messages = [{"role": "system", "content": system_prompt}]

        # 载入短期记忆
        for m in self.short_memory.to_messages():
            messages.append(m)
        messages.append({"role": "user", "content": task})

        # --- ReAct 循环 ---
        consecutive_errors = 0
        max_consecutive_errors = 2
        repeated_actions: Dict[str, int] = {}
        max_repeated_actions = 2

        for step in range(1, max_steps + 1):
            _steps_used = step
            # 检查上下文是否需要压缩
            if enable_context:
                action = self.context_manager.decide_action()
                # K5-H3: offload 在 "offload" 或 "both" 时触发
                if action in ("offload", "both"):
                    self.context_manager.auto_offload_large_results()
                # 修复：compress 分支原来空操作——现在实际执行压缩
                if action in ("compress", "both"):
                    self.context_manager.clear_compressed()

            # Token 预算强制执行——替代原来的条数截断
            # 渐进式三级压缩：清除旧工具结果 → 截断长消息 → 丢弃最旧消息
            pre_tokens = self._estimate_messages_tokens(messages)
            messages = self._enforce_token_budget(messages)
            post_tokens = self._estimate_messages_tokens(messages)
            if post_tokens < pre_tokens:
                log.info("context_compacted",
                         data={"before_tokens": pre_tokens,
                               "after_tokens": post_tokens, "step": step})

            # 流式获取 LLM 响应（含上下文超长重试）
            response = None
            for _retry in range(2):  # 最多重试 1 次
                try:
                    with metrics.timer("llm_latency", tags={"model": getattr(self.llm, "model", "unknown")}):
                        response = self._stream_llm_response(messages, stream_callback)
                    break
                except Exception as exc:
                    if _retry == 0 and self._is_context_length_error(exc):
                        # 上下文超长：激进压缩后重试
                        log.info("context_length_retry", data={"step": step, "error": str(exc)[:300]})
                        messages = self._enforce_token_budget(
                            messages, max_tokens=int(self.max_context_tokens * 0.6)
                        )
                        continue
                    # 非上下文超长错误，或重试后仍失败——终止
                    log.info("react_stream_llm_error", data={"step": step, "error": str(exc)[:500]})
                    metrics.gauge("react_steps", _steps_used)
                    # 增强错误诊断：上下文超长时给出可操作建议
                    err_detail = str(exc)
                    if self._is_context_length_error(exc):
                        err_detail += (
                            "\n\n【诊断】后端上下文窗口过小，请求 token 数超过模型限制。"
                            "\n建议:"
                            "\n  1. 启动 vLLM 时增加 --max-model-len 8192 (或更大)"
                            "\n  2. 或设置环境变量 AEROSPACE_MAX_CONTEXT_TOKENS=4096 以适配当前窗口"
                            "\n  3. 或设置 AEROSPACE_MAX_OUTPUT_TOKENS=512 减小输出预留"
                        )
                    return (
                        f"LLM 调用失败,终止推理。\n"
                        f"步骤: {step}/{max_steps}\n"
                        f"错误: {err_detail}\n"
                        f"已保留上下文与工具记录,请检查模型服务请求限制、上下文长度或服务端日志。"
                    )
            if response is None:
                response = ""
            metrics.inc("llm_calls")

            if not response.strip():
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    log.info("react_stream_end", data={"steps": _steps_used, "result_length": 0})
                    metrics.gauge("react_steps", _steps_used)
                    return "连续多次获得空响应,终止推理。"
                messages.append({"role": "assistant", "content": "(空响应)"})
                continue

            self.short_memory.add("assistant", response)
            if enable_context:
                self.context_manager.add_message("assistant", response)

            # 解析 Final Answer
            final = self._parse_final_answer(response)
            if final is not None:
                if enable_context:
                    self.context_manager.add_essential(f"Final Answer: {final}")
                self._persist_memory(task, final)
                # MemoryManager: 结束任务,保存到长期记忆
                if self.memory_manager is not None:
                    try:
                        self.memory_manager.set_working("final_answer", final)
                        self.memory_manager.end_task(save=True)
                    except Exception:
                        pass
                log.info("react_stream_end", data={"steps": _steps_used, "result_length": len(final) if final else 0})
                metrics.gauge("react_steps", _steps_used)
                return final

            # 解析 Action
            action = self._parse_action(response)
            if action is None:
                # 无 Action 也无 Final Answer,视为最终答案
                self._persist_memory(task, response)
                if self.memory_manager is not None:
                    try:
                        self.memory_manager.end_task(save=True)
                    except Exception:
                        pass
                log.info("react_stream_end", data={"steps": _steps_used, "result_length": len(response) if response else 0})
                metrics.gauge("react_steps", _steps_used)
                return response

            tool_name = action["tool"]
            args = action["args"]

            # 智能解析参数
            if isinstance(args, str):
                args = self._parse_action_input(args)

            action_signature = self._action_signature(tool_name, args)
            repeated_actions[action_signature] = repeated_actions.get(action_signature, 0) + 1
            if repeated_actions[action_signature] >= max_repeated_actions:
                log.info(
                    "react_stream_repeated_action_abort",
                    data={"tool": tool_name, "step": step, "repeat": repeated_actions[action_signature]},
                )
                metrics.gauge("react_steps", _steps_used)
                return (
                    f"重复工具调用 {repeated_actions[action_signature]} 次,终止推理。\n"
                    f"工具: {tool_name}\n"
                    f"参数: {args}\n"
                    "判定: 模型在重复同一 Action,没有消费 Observation 或改变策略。"
                )

            # 执行工具
            try:
                with metrics.timer("tool_latency", tags={"tool": tool_name}):
                    result = self._execute_tool(tool_name, args)
            except Exception as e:
                result = f"工具执行异常: {e}"
                consecutive_errors += 1
            # K5-缺陷7: 修复 metrics 状态误报——result 恒非空，用内容判定
            _tool_ok = not any(kw in str(result) for kw in ("错误", "error", "异常", "Error"))
            metrics.inc("tool_calls", tags={"tool": tool_name, "status": "success" if _tool_ok else "error"})
            log.info("tool_executed", data={"tool": tool_name, "status": "success" if _tool_ok else "error"})

            if enable_context:
                self.context_manager.add_tool_record(tool_name, args, result)
            self.short_memory.add("tool", result)

            # MemoryManager: 存入工作记忆
            if self.memory_manager is not None:
                try:
                    self.memory_manager.set_working(
                        f"step_{step}_{tool_name}", result
                    )
                except Exception:
                    pass

            # 构建 Observation（工具结果截断后追加，防止单条消息撑爆上下文）
            result = self._truncate_tool_result(result, self.tool_result_max_chars)
            obs = f"Observation: {result}"
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs})

            # 错误恢复：强化 Observation，让 LLM 明确知道必须换策略
            if "错误" in result or "error" in result.lower() or "[TOOL ERROR]" in result:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    log.info("react_stream_end", data={"steps": _steps_used, "result_length": len(result) if result else 0})
                    metrics.gauge("react_steps", _steps_used)
                    return (
                        f"连续 {max_consecutive_errors} 次工具错误,终止推理。\n"
                        f"最后结果: {result}\n"
                        f"建议: 请检查工具名和参数是否正确，或调用 list_tools 查看可用工具。"
                    )
                # 注入强化错误提示——不截断，完整传递错误信息
                error_hint = (
                    f"[系统提示] 上一步工具调用失败，这是第 {consecutive_errors} 次失败。\n"
                    f"错误详情:\n{result}\n\n"
                    f"你必须立即换用其他工具或调整参数。"
                    f"如果同一工具再次失败，推理将被强制终止。"
                )
                messages.append({"role": "user", "content": error_hint})
            else:
                consecutive_errors = 0

        log.info("react_stream_end", data={"steps": _steps_used, "result_length": 0})
        metrics.gauge("react_steps", _steps_used)
        return f"已达到最大推理步数({max_steps}),任务未得出最终答案。"

    def _stream_llm_response(
        self,
        messages: List[Dict],
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """获取 LLM 响应,支持流式回调。"""
        if stream_callback and hasattr(self.llm, "stream_chat"):
            chunks = []
            for chunk in self.llm.stream_chat(messages):
                chunks.append(chunk)
                stream_callback(chunk)
            return "".join(chunks)
        return self.llm.chat(messages)

    @staticmethod
    def _format_blueprint(bp: Dict) -> str:
        """格式化蓝图为可读文本。"""
        lines = []
        if "architecture" in bp:
            lines.append(f"架构: {bp['architecture']}")
        if "data_model" in bp:
            lines.append(f"数据模型: {bp['data_model']}")
        if "workflow_shape" in bp:
            lines.append(f"工作流形态: {bp['workflow_shape']}")
        if "key_principles" in bp:
            lines.append("关键原理:")
            for p in bp["key_principles"]:
                lines.append(f"  - {p}")
        if "risk_mitigations" in bp:
            lines.append("风险缓解:")
            for r in bp["risk_mitigations"]:
                lines.append(f"  - {r}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 默认装配工厂
# ----------------------------------------------------------------------
def _load_physics_tools() -> List[Tool]:
    """lazy import 物理模块，注册可用的物理工具（不可用则返回空）。"""
    out: List[Tool] = []
    try:
        import numpy as _np  # noqa: WPS433
        from ..physics import propagate_two_body  # noqa: WPS433

        def _two_body(r0, v0, mu, dt):
            r0 = _np.asarray(r0, dtype=float).ravel().tolist()
            v0 = _np.asarray(v0, dtype=float).ravel().tolist()
            r_out, v_out = propagate_two_body(r0, v0, float(mu), float(dt))
            r_out = _np.asarray(r_out).ravel().tolist()
            v_out = _np.asarray(v_out).ravel().tolist()
            return (f"二体传播 dt={dt}s -> r="
                    f"{[round(x, 3) for x in r_out]}, "
                    f"v={[round(x, 6) for x in v_out]}")

        out.append(Tool("two_body_propagate",
                        "二体轨道传播(r0,v0,mu,dt)", _two_body))
    except Exception:
        # 物理模块不可用时静默回退
        pass
    return out


def _load_mcp_tools() -> List[Any]:
    """lazy import MCP 工具模块，返回 BaseTool 实例列表（不可用则返回空）。

    工具体系收敛后只加载统一桥接器:
      - astro_dynamics_tool (AstroDynamicsMCPTool: 12 MCP 工具)
    LoopEngineTool 已移除——八阶段编排由 run_fast() + BaseWorkflow 替代。
    旧版单引擎工具 (orekit/gmat/spiceypy/astropy/basilisk/stk) 已由
    AstroDynamicsMCPTool 统一桥接,不再单独加载,减少维护面。
    """
    out: List[Any] = []
    try:
        import importlib  # noqa: WPS433
        from ..mcp_tools.base import BaseTool  # noqa: WPS433

        # 只加载统一桥接器 (AstroDynamicsMCPTool)
        for mod_name in ("astro_dynamics_tool",):
            try:
                mod = importlib.import_module(
                    f"aerospace_agent.mcp_tools.{mod_name}")
            except Exception:
                continue
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (isinstance(obj, type) and issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and obj.__module__ == mod.__name__):
                    try:
                        inst = obj()
                        if getattr(inst, "name", None):
                            out.append(inst)
                    except Exception:
                        continue
    except Exception:
        pass
    return out


def create_default_agent(max_steps: int = 10,
                         force_mock: bool = False,
                         use_local: bool = False,
                         use_router: bool = False) -> AerospaceAgent:
    """工厂函数：自动装配所有默认组件。

    会 lazy import 物理模块、MCP 工具、工作流、RAG：
      - LLM：默认 MockLLM；use_router=True 自动路由本地/云端；
        use_local=True 或设置 AEROSPACE_LOCAL_LLM_BASE_URL 用本地小模型；
      - MCP 工具：astro_dynamics_mcp 12 工具（LoopEngineTool 已移除）；
      - BaseWorkflow：workflows/ 包的 5 个预定义工作流，run_fast() 优先匹配；
      - RAG：AerospaceRAG（增强版，含 RetrieverRouter 多源路由 +
        EvidenceVerifier 证据验证 + TraceabilityManager 溯源链）；
      - 工作流/物理工具使用内置默认实现。
    所有外部模块均 lazy import，不可用时静默回退到内置实现。
    """
    # 1. LLM（无 API key 时自动回退 MockLLM；支持本地小模型 + 路由器）
    llm = create_llm(force_mock=force_mock,
                     use_local=use_local, use_router=use_router)
    # 2. 上下文管理器
    ctx = ContextManager()
    # 3. 记忆
    memory = LongTermMemory()
    # 4. 工具：内置默认 + lazy 物理模块
    tools = default_tools()
    tools.extend(_load_physics_tools())
    # 5. 工作流：内置默认
    workflows = default_workflows()
    # 6. 组装 Agent
    agent = AerospaceAgent(
        llm=llm, context_manager=ctx, memory=memory,
        tools=tools, workflows=workflows, max_steps=max_steps,
    )
    # 7. lazy 加载 MCP 工具（astro_dynamics 12 工具，LoopEngineTool 已移除）
    for bt in _load_mcp_tools():
        agent.register_mcp_tool(bt)
    # 7.5 注册 BaseWorkflow 实例（workflows/ 包的 5 个预定义工作流）
    #     这些工作流支持 execute() 直接执行，run_fast() 优先匹配并调用
    try:
        from ..workflows.registry import default_workflow_registry
        for wf_name, wf_instance in default_workflow_registry.list_all().items():
            agent.register_base_workflow(wf_instance)
    except Exception as exc:
        _logger.warning("步骤7.5 BaseWorkflow 注册失败: %s", exc)
    # 8. 挂载增强 RAG（向量库 + 关键词 + 知识图谱 + 文献管线 + 知识云图
    #    + RetrieverRouter 多源路由 + EvidenceVerifier 证据验证 + Traceability 溯源）
    try:
        from ..rag.aerospace_rag import AerospaceRAG
        agent.rag = AerospaceRAG()
        # K2.5: 注册多源检索器——记忆源 + 代码源
        # 记忆检索源：将 LongTermMemory.recall 包装为 (query, top_k) -> [(score, text, meta)]
        def _memory_retriever(query: str, top_k: int = 5):
            results = memory.recall(query, top_k=top_k)
            return [
                (sim, str(val)[:500], {"key": key, "source_type": "memory"})
                for key, sim, val in results
            ]
        agent.rag.register_memory_retriever(_memory_retriever)

        # 代码检索源：搜索项目源码中匹配的函数/类/注释
        def _code_retriever(query: str, top_k: int = 5):
            import ast as _ast
            import glob as _glob
            import os as _os
            # 限定搜索范围：aerospace_agent 包目录
            pkg_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            py_files = _glob.glob(_os.path.join(pkg_dir, "**", "*.py"), recursive=True)
            scored = []
            query_lower = query.lower()
            for fpath in py_files:
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                    tree = _ast.parse(source)
                except Exception:
                    continue
                rel_path = _os.path.relpath(fpath, pkg_dir)
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                        name = node.name
                        docstring = _ast.get_docstring(node) or ""
                        # 简单关键词匹配
                        score = 0.0
                        for word in query_lower.split():
                            if len(word) > 1:
                                if word in name.lower():
                                    score += 2.0
                                if word in docstring.lower():
                                    score += 1.0
                        for kw in query.split():
                            if len(kw) > 1:
                                if kw in name:
                                    score += 0.3
                        if score > 0:
                            snippet = f"def {name}(...)" if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) else f"class {name}"
                            scored.append((
                                score,
                                f"{rel_path}:{node.lineno} {snippet}\n{docstring[:200]}",
                                {"file": rel_path, "line": node.lineno,
                                 "source_type": "code"},
                            ))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        agent.rag.register_code_retriever(_code_retriever)
    except Exception:
        # 回退到简易 RAG
        agent.rag = SimpleRAG(memory)
    # 9. 挂载 Skill 注册表（自动发现所有内置 Skill）
    try:
        from ..skills import SkillRegistry
        from ..skills.defaults import install_default_skill_manifests
        agent.skills = SkillRegistry()
        agent.skills.auto_discover()
        install_default_skill_manifests(agent.skills)
    except Exception as exc:
        _logger.warning("步骤9 SkillRegistry 挂载失败: %s", exc)
    # 9.5 挂载 MemoryManager (统一管理三层记忆,接入主流程)
    try:
        agent.memory_manager = MemoryManager(
            long_term_path=str(memory.path) if hasattr(memory, 'path') else None,
        )
        # 让 MemoryManager 共享已有的 LongTermMemory
        agent.memory_manager.long_term = memory
    except Exception as exc:
        _logger.warning("步骤9.5 MemoryManager 挂载失败: %s", exc)
    # 10. 挂载 Prompt 模板
    try:
        from ..prompts import SYSTEM_PROMPT, get_prompt
        agent.prompt_template = get_prompt
        agent.system_prompt = SYSTEM_PROMPT
    except Exception as exc:
        _logger.warning("步骤10 Prompt 模板挂载失败: %s", exc)
    # 11. 注册文献搜索工具（让 Agent 在 ReAct 循环中可调用）
    try:
        def _literature_search(query: str, topic: str = "", max_results: int = 5) -> str:
            """搜索最新航天文献，评估相关性，下载强相关论文并总结。"""
            rag_obj = getattr(agent, "rag", None)
            if rag_obj is None or not hasattr(rag_obj, "search_literature"):
                return "RAG 文献管线不可用"
            result = rag_obj.search_literature(
                query=query, research_topic=topic or query,
                max_results=max_results, download_strong=True,
            )
            lines = [
                f"搜索到 {result['total_found']} 篇，"
                f"强相关 {result['strong_count']} 篇（已下载 {result['downloaded_count']} 篇），"
                f"弱相关 {result['weak_count']} 篇（已跳过）。",
            ]
            for p in result.get("papers", []):
                if p["relevance"] == "strong":
                    lines.append(f"  [{p['status']}] {p['title']}")
                    if p["summary"]:
                        lines.append(f"    总结: {p['summary']}")
            snap = result.get("knowledge_graph_snapshot", {})
            if snap:
                lines.append(f"知识图谱: {snap.get('nodes','?')} 节点, {snap.get('edges','?')} 边")
            return "\n".join(lines)
        agent.register_tool(Tool(
            "literature_search",
            "搜索最新航天文献(query, topic, max_results)：评估相关性、下载强相关 PDF、总结全文、更新知识图谱",
            _literature_search,
        ))
    except Exception as exc:
        _logger.warning("步骤11 文献搜索工具注册失败: %s", exc)

    # 12. 注册 105 个科研工具（research_tools 包）
    #     10 个域 × 10-15 个原子操作 = 最小生成集
    #     支持自进化：工具不存在时 Agent 可用 create_tool 动态创建
    try:
        from ..research_tools import (
            get_registry as _get_rt_registry,
            get_all_schemas as _get_rt_schemas,
        )
        _rt_registry = _get_rt_registry()
        for tool_name in _rt_registry.list_all():
            rt = _rt_registry.get(tool_name)
            if rt is None:
                continue
            # 包装为 Agent Tool（统一接口）
            def _make_rt_caller(name):
                def _caller(**kwargs):
                    return _get_rt_registry().call(name, **kwargs)
                _caller.__name__ = f"rt_{name}"
                return _caller
            agent.register_tool(Tool(
                tool_name,
                rt.to_schema(),
                _make_rt_caller(tool_name),
            ))
        _logger.info("步骤12: 注册 %d 个科研工具", _rt_registry.count())

        # 13. 构建工具向量索引——语义检索 top-K 工具，替代全量注入
        #     105 个工具全量注入 ~2015 tokens → top-8 精选 ~300 tokens
        try:
            import os as _os
            from ..research_tools.tool_index import ToolVectorIndex
            _idx_path = _os.path.join(_os.getcwd(), "data", "tool_index.npz")
            agent.tool_index = ToolVectorIndex(persist_path=_idx_path)
            if not agent.tool_index.is_built:
                agent.tool_index.build_from_registry(_rt_registry)
            _logger.info("步骤13: 工具向量索引构建完成 (%d 个工具)",
                         agent.tool_index.get_stats()["total_tools"])
        except Exception as exc:
            _logger.warning("步骤13 工具向量索引构建失败: %s", exc)

    except Exception as exc:
        _logger.warning("步骤12 科研工具注册失败: %s", exc)

    return agent
