"""递归第一性原理分析器 — LoopRecursive-CEO Phase A。

将用户需求递归下钻到第一性原理：
  1. 提取需求关键词，按"错误代价"评分
  2. Top-K 排序（K=3-7），按上下文链接保持设计连贯
  3. 每个分支递归到可构建的第一性原理决策
  4. 自底向上综合为 v1 蓝图

这是 Plan 阶段的核心——前置深度思考，使后续 Loop 快速收敛。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DecisionNode:
    """递归决策树的节点。

    Attributes:
        keyword: 需求关键词
        score: 错误代价评分（0-10，越高越关键）
        sub_meaning_a / sub_meaning_b: 二叉拆分的两个子含义
        first_principle: 下钻到的第一性原理（叶节点）
        children: 子节点
        context_link: 与更高优先级节点的上下文链接说明
    """
    keyword: str = ""
    score: float = 0.0
    sub_meaning_a: str = ""
    sub_meaning_b: str = ""
    first_principle: str = ""
    children: List["DecisionNode"] = field(default_factory=list)
    context_link: str = ""

    def is_leaf(self) -> bool:
        """是否为叶节点（已下钻到第一性原理）。"""
        return bool(self.first_principle) and not self.children

    def to_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "score": self.score,
            "sub_meaning_a": self.sub_meaning_a,
            "sub_meaning_b": self.sub_meaning_b,
            "first_principle": self.first_principle,
            "children": [c.to_dict() for c in self.children],
            "context_link": self.context_link,
        }


class FirstPrinciplesAnalyzer:
    """递归第一性原理分析器。

    用法：
        analyzer = FirstPrinciplesAnalyzer(llm=...)
        blueprint = analyzer.analyze(
            goal="设计地月转移轨道",
            constraints=["无网络环境", "精度<1km"],
            context={...},
        )
    """

    # Top-K 参数：保留最高优先级的 K 个铰链决策
    TOP_K: int = 5
    # 最大递归深度
    MAX_DEPTH: int = 4

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选的 LLM 接口（LLMInterface）。无 LLM 时用规则回退。
        """
        self.llm = llm

    def analyze(self, goal: str, constraints: List[str] = None,
                context: Dict[str, Any] = None) -> dict:
        """执行递归第一性原理分析，返回 v1 蓝图。

        Returns:
            {
                "goal": 原始目标,
                "constraints": 约束列表,
                "top_k_nodes": [DecisionNode.to_dict()...],  # Top-K 铰链决策
                "blueprint": {  # 综合 v1 蓝图
                    "architecture": 架构决策,
                    "data_model": 数据模型决策,
                    "workflow_shape": 工作流形态,
                    "risk_mitigations": 风险缓解,
                },
                "decision_ledger": [决策记录...],
            }
        """
        constraints = constraints or []
        context = context or {}

        # 1. 提取并评分关键词
        keywords = self._extract_keywords(goal, constraints, context)
        scored = self._score_keywords(keywords, goal, context)

        # 2. Top-K 排序
        top_k = sorted(scored, key=lambda x: x[1], reverse=True)[:self.TOP_K]

        # 3. 递归下钻每个 Top-K 节点
        nodes: List[DecisionNode] = []
        for kw, score, sub_a, sub_b in top_k:
            node = DecisionNode(keyword=kw, score=score,
                                sub_meaning_a=sub_a, sub_meaning_b=sub_b)
            self._recurse(node, depth=0, context=context, prior_nodes=nodes)
            nodes.append(node)

        # 4. 上下文链接（让设计保持连贯）
        self._link_context(nodes)

        # 5. 自底向上综合蓝图
        blueprint = self._synthesize(nodes, goal, constraints)

        return {
            "goal": goal,
            "constraints": constraints,
            "top_k_nodes": [n.to_dict() for n in nodes],
            "blueprint": blueprint,
            "decision_ledger": self._build_ledger(nodes),
            "timestamp": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # 关键词提取与评分
    # ------------------------------------------------------------------
    def _extract_keywords(self, goal: str, constraints: List[str],
                          context: Dict) -> List[str]:
        """从目标、约束、上下文中提取需求关键词。"""
        text = goal + " " + " ".join(constraints)
        # 航天领域关键词词典
        domain_keywords = [
            "轨道", "传播", "propagat", "转移", "transfer", "发射窗口",
            "launch", "姿态", "attitude", "可见性", "access", "地面站",
            "坐标系", "frame", "转换", "transform", "星历", "ephemeris",
            "引力", "gravity", "球谐", "spherical", "阻力", "drag",
            "光压", "srp", "三体", "third_body", "变轨", "maneuver",
            "霍曼", "hohmann", "Lambert", "验证", "validate", "精度",
            "精度要求", "二体", "two_body", "数值", "numerical",
            "SGP4", "TLE", "历元", "epoch", "时间尺度", "time_scale",
        ]
        found = []
        low = text.lower()
        for kw in domain_keywords:
            if kw.lower() in low:
                found.append(kw)
        # 如果有 LLM，让它补充提取
        if self.llm and len(found) < 3:
            try:
                resp = self.llm.chat([
                    {"role": "system", "content": "提取航天任务需求关键词，逗号分隔。"},
                    {"role": "user", "content": text},
                ])
                found.extend([k.strip() for k in resp.split(",") if k.strip()])
            except Exception:
                pass
        return list(dict.fromkeys(found)) or ["general_task"]

    def _score_keywords(self, keywords: List[str], goal: str,
                        context: Dict) -> List[Tuple[str, float, str, str]]:
        """评分并二叉拆分关键词。

        Returns:
            [(keyword, score, sub_meaning_a, sub_meaning_b), ...]
        """
        # 错误代价评分矩阵（错误代价越高，分数越高）
        cost_matrix = {
            "轨道": 9, "传播": 9, "propagat": 9,
            "转移": 10, "transfer": 10,
            "坐标系": 8, "frame": 8, "转换": 8, "transform": 8,
            "历元": 7, "epoch": 7, "时间尺度": 7, "time_scale": 7,
            "发射窗口": 8, "launch": 8,
            "可见性": 6, "access": 6, "地面站": 6,
            "姿态": 7, "attitude": 7,
            "引力": 7, "gravity": 7, "球谐": 8, "spherical": 8,
            "阻力": 5, "drag": 5, "光压": 5, "srp": 5,
            "变轨": 8, "maneuver": 8,
            "验证": 6, "validate": 6, "精度": 7,
            "二体": 6, "two_body": 6,
            "数值": 5, "numerical": 5,
            "星历": 7, "ephemeris": 7,
            "Lambert": 8, "霍曼": 7, "hohmann": 7,
        }
        # 二叉拆分模板
        split_templates = {
            "轨道": ("轨道模型选择（二体/数值/SGP4）", "轨道表示（笛卡尔/开普勒）"),
            "传播": ("传播器类型（解析/数值）", "积分器选择（RK4/DP853）"),
            "转移": ("能量需求（C3/Δv）", "转移类型（Hohmann/Lambert/拼凑圆锥）"),
            "坐标系": ("源系定义（GCRF/ITRF）", "转换方法（极运动/岁差章动）"),
            "历元": ("时间尺度（UTC/TDB）", "格式（ISO/JD/MJD）"),
            "验证": ("验证基准（多引擎交叉）", "容差阈值（位置/速度）"),
        }
        result = []
        for kw in keywords:
            score = cost_matrix.get(kw.lower(), 3.0)
            sub_a, sub_b = split_templates.get(kw, (f"{kw}的物理约束", f"{kw}的工程选择"))
            result.append((kw, score, sub_a, sub_b))
        return result

    # ------------------------------------------------------------------
    # 递归下钻
    # ------------------------------------------------------------------
    def _recurse(self, node: DecisionNode, depth: int,
                 context: Dict, prior_nodes: List[DecisionNode]) -> None:
        """递归下钻到第一性原理。"""
        if depth >= self.MAX_DEPTH:
            node.first_principle = self._derive_principle(node, context)
            return

        # 拆分为两个子节点
        child_a = DecisionNode(keyword=node.sub_meaning_a, score=node.score * 0.8)
        child_b = DecisionNode(keyword=node.sub_meaning_b, score=node.score * 0.8)

        # 进一步拆分子节点
        for child in (child_a, child_b):
            if self._is_buildable(child, context):
                child.first_principle = self._derive_principle(child, context)
            else:
                self._recurse(child, depth + 1, context, prior_nodes)
            node.children.append(child)

        # 如果子节点都已下钻，当前节点的第一性原理 = 子节点综合
        if all(c.first_principle for c in node.children):
            node.first_principle = " + ".join(
                c.first_principle for c in node.children
            )

    def _is_buildable(self, node: DecisionNode, context: Dict) -> bool:
        """判断节点是否已达到可构建的第一性原理（叶节点）。"""
        # 简单启发式：如果关键词包含具体选择，则为叶节点
        buildable_signals = ["选择", "类型", "方法", "基准", "阈值",
                             "定义", "格式", "约束", "需求"]
        return any(sig in node.keyword for sig in buildable_signals) or not node.keyword

    def _derive_principle(self, node: DecisionNode, context: Dict) -> str:
        """推导第一性原理。

        如果有 LLM，调用 LLM 深度推理；否则用规则模板。
        """
        if self.llm:
            try:
                resp = self.llm.chat([
                    {"role": "system", "content": (
                        "你是航天动力学架构师。对以下决策关键词，用一句话给出其第一性原理"
                        "（不可再下钻的物理或工程基石）。"
                    )},
                    {"role": "user", "content": node.keyword or node.sub_meaning_a},
                ])
                return resp.strip()
            except Exception:
                pass
        # 规则回退
        templates = {
            "轨道模型选择": "物理状态必须 SI 单位 + epoch/frame 标签，引擎无关",
            "轨道表示": "同一状态可笛卡尔/开普勒表示，必须显式声明且无损往返",
            "传播器类型": "解析传播快但精度低，数值传播精确但耗算力，按精度需求选择",
            "积分器选择": "DP853 自适应步长兼顾精度与效率，RK4 简单但精度低",
            "能量需求": "C3 和 Δv 由能量守恒唯一确定，转移类型决定计算方法",
            "转移类型": "Hohmann 最低能量但约束共面圆轨道，Lambert 通用但需解两点边值",
            "源系定义": "惯性系(GCRF)用于轨道动力学，固连系(ITRF)用于地面定位",
            "转换方法": "GCRF→ITRF 需极运动+地球自转+岁差章动，精度依赖 IERS 数据",
            "时间尺度": "UTC 有闰秒，TDB 无闰秒但需相对论修正，跨尺度转换必须显式",
            "格式": "ISO 人类可读，JD/MJD 机器友好，跨格式必须保持精度",
            "验证基准": "多引擎交叉验证是可信度的唯一来源，无基准则不可信",
            "容差阈值": "位置误差 <100m 为高可信，<1km 为中可信，>1km 需排查",
        }
        for key, principle in templates.items():
            if key in (node.keyword or "") or key in (node.sub_meaning_a or ""):
                return principle
        return f"{node.keyword or node.sub_meaning_a}: 需明确物理约束与工程选择的权衡"

    # ------------------------------------------------------------------
    # 上下文链接与综合
    # ------------------------------------------------------------------
    def _link_context(self, nodes: List[DecisionNode]) -> None:
        """为每个节点添加与更高优先级节点的上下文链接。"""
        for i, node in enumerate(nodes):
            if i == 0:
                node.context_link = "最高优先级：整个系统 hinges on 此决策"
            else:
                prior = nodes[i - 1]
                node.context_link = (
                    f"在「{prior.keyword}」已决策的基础上，本节点须保持一致："
                    f"不违背 {prior.first_principle[:30]}..."
                )

    def _synthesize(self, nodes: List[DecisionNode], goal: str,
                    constraints: List[str]) -> dict:
        """自底向上综合为 v1 蓝图。"""
        principles = [n.first_principle for n in nodes if n.first_principle]
        return {
            "architecture": (
                "Canonical Astrodynamics Model 统一中间层 + 7 引擎 Adapter + "
                "Loop 八阶段编排 + ReAct 步骤执行"
            ),
            "data_model": (
                "所有物理量 SI 单位显式标注 + epoch/frame 标签 + "
                "可序列化 JSON + canonical→engine→canonical 无损往返"
            ),
            "workflow_shape": (
                "Plan(递归第一性原理) → SelectEngine(能力匹配) → "
                "RetrieveDemo(索引搜索) → GenerateWorkflow(YAML) → "
                "Run(ReAct执行) → Validate(多引擎交叉) → Fix(最小修复) → "
                "Save(沉淀可复用)"
            ),
            "risk_mitigations": [
                "引擎不可用 → 结构化 unavailable，绝不崩溃",
                "精度不足 → 多引擎交叉验证 + 可信度评级",
                "约束冲突 → 递归分析阶段识别并升级",
            ],
            "key_principles": principles,
        }

    def _build_ledger(self, nodes: List[DecisionNode]) -> List[dict]:
        """构建决策账本。"""
        ledger = []
        for i, node in enumerate(nodes):
            ledger.append({
                "rank": i + 1,
                "keyword": node.keyword,
                "score": node.score,
                "first_principle": node.first_principle,
                "context_link": node.context_link,
            })
        return ledger
