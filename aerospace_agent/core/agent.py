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

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .context_manager import ContextManager
from .llm_interface import LLMInterface, create_llm
from .memory import LongTermMemory, ShortTermMemory


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
    """安全表达式计算器（仅允许数字与数学运算符）。"""
    if not re.fullmatch(r"[0-9eE+\-*/()., ]+", expression):
        return "错误：表达式包含非法字符"
    try:
        value = eval(expression, {"__builtins__": {}}, {"math": math})
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
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.workflows: Dict[str, Workflow] = {w.name: w for w in (workflows or [])}
        self.max_steps = max_steps
        # MCP 工具注册表（BaseTool 实例，接口与原生 Tool 不同，单独存放）
        self.mcp_tools: Dict[str, Any] = {}
        # 可选挂载的 RAG（由 create_default_agent 设置）
        self.rag: Optional[SimpleRAG] = None

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

    # ------------------------------------------------------------------
    # ReAct 解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_action(text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 输出中解析 Action / Action Input。"""
        action_match = re.search(r"Action:\s*(.+)", text)
        if not action_match:
            return None
        tool_name = action_match.group(1).strip()
        input_match = re.search(r"Action Input:\s*(.+)", text)
        raw_input = input_match.group(1).strip() if input_match else "{}"
        # 尝试解析为 JSON；失败则把整段作为 input 参数
        try:
            args = json.loads(raw_input)
            if not isinstance(args, dict):
                args = {"input": args}
        except json.JSONDecodeError:
            args = {"input": raw_input}
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
        """执行指定工具（原生 Tool 或 MCP BaseTool），返回结果字符串。"""
        # 原生工具
        tool = self.tools.get(tool_name)
        if tool is not None:
            try:
                result = tool(**args) if isinstance(args, dict) else tool(args)
                return str(result)
            except Exception as e:
                return f"工具执行错误: {e}"
        # MCP 工具（BaseTool.call(method, **kwargs) 接口）
        bt = self.mcp_tools.get(tool_name)
        if bt is not None:
            try:
                kw = dict(args) if isinstance(args, dict) else {"input": args}
                method = kw.pop("method", None)
                if method is None:
                    methods = getattr(bt, "list_methods", lambda: [])()
                    method = methods[0] if methods else ""
                res = bt.call(method, **kw)
                return json.dumps(res, ensure_ascii=False, default=str)
            except Exception as e:
                return f"MCP 工具执行错误: {e}"
        return f"错误：未知工具 '{tool_name}'"

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
            step_messages = list(messages)
            if observation:
                step_messages.append({"role": "user", "content": observation})

            try:
                response = self.llm.chat(step_messages)
            except Exception as e:
                print(f"[LLM 调用失败] {e}")
                return f"任务失败：LLM 调用异常 - {e}"

            # 打印思考（截断过长输出）
            preview = response if len(response) <= 500 else response[:500] + " ..."
            print(f"[Thought]\n{preview}")
            self.short_memory.add("assistant", response)
            self.context_manager.add_message("assistant", response)

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
            print(f"[Observation] {result}")
            self.context_manager.add_tool_record(tool_name, args, result)
            self.short_memory.add("tool", result)
            observation = f"Observation: {result}"

        print(f"\n[达到最大步数 {self.max_steps}，终止循环]")
        print("========== 任务结束 ==========\n")
        return "已达到最大推理步数，任务未得出最终答案。"

    def _persist_memory(self, task: str, answer: str) -> None:
        """将任务与答案持久化到长期记忆。"""
        try:
            self.memory.remember(task, answer, tags=["task_result"])
            self.memory.save()
        except Exception as e:
            print(f"[记忆持久化失败] {e}")


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
    """lazy import MCP 工具模块，返回 BaseTool 实例列表（不可用则返回空）。"""
    out: List[Any] = []
    try:
        import importlib  # noqa: WPS433
        from ..mcp_tools.base import BaseTool  # noqa: WPS433
        for mod_name in ("astropy_tool", "gmat_tool", "orekit_tool",
                         "spiceypy_tool"):
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
        # MCP 模块不可用时静默回退
        pass
    return out


def create_default_agent(max_steps: int = 10,
                         force_mock: bool = False) -> AerospaceAgent:
    """工厂函数：自动装配所有默认组件。

    会 lazy import 物理模块、MCP 工具、工作流、RAG：
      - 物理模块（aerospace_agent.physics）可用则注册 two_body_propagate 工具；
      - MCP 工具（aerospace_agent.mcp_tools）可用则注册（含可用性检测）；
      - 工作流/RAG 使用内置默认实现。
    所有外部模块均 lazy import，不可用时静默回退到内置实现。
    """
    # 1. LLM（无 API key 时自动回退 MockLLM）
    llm = create_llm(force_mock=force_mock)
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
    # 7. lazy 加载 MCP 工具
    for bt in _load_mcp_tools():
        agent.register_mcp_tool(bt)
    # 8. 挂载增强 RAG（向量库 + 关键词 + 知识图谱 + 文献管线 + 知识云图）
    try:
        from ..rag.aerospace_rag import AerospaceRAG
        agent.rag = AerospaceRAG()
    except Exception:
        # 回退到简易 RAG
        agent.rag = SimpleRAG(memory)
    # 9. 注册文献搜索工具（让 Agent 在 ReAct 循环中可调用）
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
    except Exception:
        pass
    return agent
