"""RetrieverRouter — 多源检索路由器。

第一性原理（K5）：
    RAG 不应该只是"向量数据库 + 大模型"，而应该是
    "可路由、可检索、可验证、可追踪的知识工具系统"。

RetrieverRouter 是 RAG 系统的入口——根据查询意图路由到不同检索源：
    - 文档检索：项目文档、PDF、Markdown、Word（用现有 HybridRetriever）
    - 数据库检索：业务表、用户记录、订单、日志
    - 代码检索：代码库、接口、README
    - 记忆检索：用户偏好、历史决策（用现有 LongTermMemory）
    - 网络搜索：需要最新信息时使用

每个源各自做 Hybrid Search（向量 + 关键词），结果统一归并、去重、溯源。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .retriever import HybridRetriever, RetrievalResult

# 模块级日志器：多源路由降级路径异常告警（替代静默 except）
_logger = logging.getLogger(__name__)


@dataclass
class RoutedResult:
    """多源路由检索结果。

    Attributes:
        query: 原始查询
        results: 检索结果列表
        sources_used: 使用的检索源列表
        routing_reason: 路由原因说明
        metadata: 额外元数据
    """
    query: str = ""
    results: List[RetrievalResult] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    routing_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [
                {"text": r.text, "score": r.score, "source": r.source,
                 "metadata": r.metadata, "explanation": r.explanation}
                for r in self.results
            ],
            "sources_used": self.sources_used,
            "routing_reason": self.routing_reason,
            "metadata": self.metadata,
        }


class RetrieverRouter:
    """多源检索路由器。

    用法：
        router = RetrieverRouter(
            doc_retriever=hybrid_retriever,
            code_retriever=code_search_fn,
            memory_retriever=memory_recall_fn,
            web_retriever=web_search_fn,
        )
        result = router.retrieve("地月转移轨道 C3 能量计算")
    """

    # 查询意图 → 检索源 映射
    INTENT_KEYWORDS = {
        "document": ["文档", "论文", "paper", "pdf", "spec", "规格", "手册",
                      "manual", "文档", "markdown", "word"],
        "database": ["订单", "记录", "日志", "数据", "表", "order", "log",
                      "record", "database", "sql", "查询数据"],
        "code": ["代码", "函数", "接口", "api", "code", "function", "class",
                  "module", "repo", "readme", "实现"],
        "memory": ["偏好", "决策", "历史", "之前", "记得", "preference",
                    "history", "decision", "上次"],
        "web": ["最新", "新闻", "news", "当前", "实时", "today", "latest",
                "2024", "2025", "2026", "update"],
    }

    def __init__(self, doc_retriever: HybridRetriever = None,
                 code_retriever: Callable[[str, int], List] = None,
                 memory_retriever: Callable[[str, int], List] = None,
                 web_retriever: Callable[[str, int], List] = None,
                 db_retriever: Callable[[str, int], List] = None):
        """
        Args:
            doc_retriever: 文档检索器（HybridRetriever 实例）
            code_retriever: 代码检索函数 (query, top_k) -> [(score, text, meta)]
            memory_retriever: 记忆检索函数 (query, top_k) -> [(score, text, meta)]
            web_retriever: 网络搜索函数 (query, top_k) -> [(score, text, meta)]
            db_retriever: 数据库检索函数 (query, top_k) -> [(score, text, meta)]
        """
        self.doc_retriever = doc_retriever
        self.code_retriever = code_retriever
        self.memory_retriever = memory_retriever
        self.web_retriever = web_retriever
        self.db_retriever = db_retriever

    def detect_intent(self, query: str) -> List[str]:
        """检测查询意图，返回匹配的检索源列表。

        Returns:
            源名列表，如 ["document", "code"]。默认 ["document"]。
        """
        low = query.lower()
        scores: Dict[str, int] = {}
        for source, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in low or kw in query)
            if score > 0:
                scores[source] = score
        if not scores:
            return ["document"]
        # 取得分最高的源，如果有多个并列，全部返回
        max_score = max(scores.values())
        return [s for s, sc in scores.items() if sc == max_score]

    def retrieve(self, query: str, top_k: int = 5,
                 force_sources: List[str] = None) -> RoutedResult:
        """多源路由检索——含降级策略。

        降级策略：
          1. 外部源（web/database）失败 → 回退到本地源（document/memory）
          2. 所有源均无结果 → 回退到内置航天知识库
          3. 降级状态记录在 metadata.degradation 中

        Args:
            query: 查询文本
            top_k: 每个源返回的最大结果数
            force_sources: 强制使用的源列表（覆盖自动检测）
        Returns:
            RoutedResult 包含所有源的合并结果
        """
        sources = force_sources or self.detect_intent(query)
        all_results: List[RetrievalResult] = []
        used: List[str] = []
        failed: List[str] = []
        fallbacks_used: List[str] = []

        # 第一轮：并行尝试意图匹配的源
        import concurrent.futures

        def _query_one(source_name: str):
            retriever = getattr(self, f"{source_name}_retriever", None)
            if retriever is None:
                return source_name, None, False
            try:
                raw = self._query_source(source_name, retriever, query, top_k)
                return source_name, raw, True
            except Exception as exc:
                _logger.warning("检索源 %s 查询失败，降级跳过: %s", source_name, exc)
                return source_name, None, False

        max_workers = min(4, len(sources)) if sources else 1
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as pool:
            futures = {
                pool.submit(_query_one, src): src for src in sources
            }
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    src, raw, ok = future.result(timeout=30)
                    if ok and raw:
                        all_results.extend(raw)
                        used.append(src)
                    else:
                        failed.append(src)
                except (concurrent.futures.TimeoutError, Exception):
                    failed.append(futures[future])

        # 第二轮：降级——外部源失败时尝试本地源
        local_sources = ["document", "memory"]
        external_failed = [s for s in failed if s not in local_sources]
        if external_failed:
            for source in local_sources:
                if source in used:
                    continue  # 已在第一轮成功
                retriever = getattr(self, f"{source}_retriever", None)
                if retriever is None:
                    continue
                try:
                    raw = self._query_source(source, retriever, query, top_k)
                    if raw:
                        all_results.extend(raw)
                        used.append(source)
                        fallbacks_used.append(source)
                except Exception as exc:
                    _logger.warning("降级检索源 %s 失败: %s", source, exc)

        # 第三轮：终极回退——所有源无结果时用内置知识
        if not all_results:
            builtin = self._builtin_fallback(query, top_k)
            if builtin:
                all_results.extend(builtin)
                fallbacks_used.append("builtin_knowledge")

        # 去重 + 合并排序
        merged = self._merge_dedupe(all_results)

        # 构建降级元数据
        degradation = None
        if failed or fallbacks_used:
            degradation = {
                "failed_sources": failed,
                "fallbacks_used": fallbacks_used,
                "is_degraded": bool(failed),
                "level": "partial" if used else "full_fallback",
            }

        routing_reason = f"查询意图匹配源: {sources}，实际可用: {used}"
        if fallbacks_used:
            routing_reason += f"，降级回退: {fallbacks_used}"

        return RoutedResult(
            query=query,
            results=merged[:top_k * 2],
            sources_used=used + fallbacks_used,
            routing_reason=routing_reason,
            metadata={
                "total_candidates": len(all_results),
                "merged_count": len(merged),
                "degradation": degradation,
            },
        )

    def _query_source(self, source: str, retriever, query: str,
                      top_k: int) -> List[RetrievalResult]:
        """查询单个检索源，统一转换为 RetrievalResult。"""
        # 文档源用 HybridRetriever（已有 RetrievalResult 输出）
        if source == "document" and isinstance(retriever, HybridRetriever):
            return retriever.retrieve(query, top_k=top_k)
        # 其他源是函数式接口 (query, top_k) -> [(score, text, meta)]
        raw = retriever(query, top_k) if callable(retriever) else []
        results = []
        for item in raw:
            if isinstance(item, RetrievalResult):
                item.metadata["source_type"] = source
                results.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                score, text = item[0], item[1]
                meta = item[2] if len(item) > 2 else {}
                meta["source_type"] = source
                results.append(RetrievalResult(
                    text=text, score=float(score),
                    source=source, metadata=meta))
        return results

    @staticmethod
    def _merge_dedupe(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """跨源去重合并（按归一化文本键）。"""
        import re as _re
        merged: Dict[str, RetrievalResult] = {}
        for r in results:
            key = _re.sub(r"[\s\W_]+", "", r.text).lower()[:200]
            if not key:
                continue
            if key not in merged:
                merged[key] = r
            else:
                # 合并：累加分数，合并源标签
                existing = merged[key]
                existing.score = max(existing.score, r.score)
                if r.source not in existing.source:
                    existing.source = f"{existing.source}+{r.source}"
                existing.metadata.update(r.metadata)
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)

    # ------------------------------------------------------------------
    # 内置知识回退
    # ------------------------------------------------------------------
    _BUILTIN_KNOWLEDGE = [
        ("开普勒第三定律：T² = (4π²/μ) × a³，其中 T 为轨道周期，a 为半长轴，μ 为中心天体引力参数。",
         "orbital_mechanics"),
        ("vis-viva 方程：v² = μ(2/r - 1/a)，用于计算轨道上任意点的速度，r 为当前距离，a 为半长轴。",
         "orbital_mechanics"),
        ("霍曼转移轨道：最省能量的双脉冲转移轨道，近日点和远日点各一次切向助推，转移角为 180°。",
         "orbital_mechanics"),
        ("J2 摄动：地球扁率引起的长期摄动，导致 RAAN 进动和近地点幅角漂移。RAAN 进动率：dΩ/dt = -3/2 × J2 × (R_E/a)² × n × cos(i)。",
         "perturbations"),
        ("拉格朗日点 L1-L5：三体问题中的五个平衡点，L1/L2/L3 不稳定，L4/L5 稳定（当质量比 > 25）。",
         "three_body"),
        ("C3 能量：C3 = v_inf²，表示逃逸双曲线剩余速度的平方，是行星际任务设计的核心能量参数。",
         "interplanetary"),
        ("轨道六根数：a（半长轴）、e（偏心率）、i（倾角）、Ω（升交点赤经）、ω（近地点幅角）、ν（真近点角）。",
         "orbital_elements"),
        ("二体问题解析解：开普勒方程 M = E - e×sin(E)，其中 M 为平近点角，E 为偏近点角，通过牛顿迭代求解。",
         "orbital_mechanics"),
        ("GCRF（地心天球参考系）：近似惯性系，原点在地心，z 轴指向 J2000 平北天极，x 轴指向 J2000 平春分点。",
         "frames"),
        ("ITRF（国际地球参考系）：固连地球的旋转坐标系，原点在地心，z 轴指向 CIO（协议国际原点）。",
         "frames"),
        ("球谐引力模型：地球引力场用球谐函数展开，J2=1.08263e-3 是最大项，表示地球扁率效应。",
         "gravity_model"),
        ("大气阻力：F_drag = -0.5 × ρ × v² × Cd × A，其中 ρ 为大气密度，Cd 为阻力系数，A 为截面积。",
         "perturbations"),
    ]

    def _builtin_fallback(self, query: str, top_k: int) -> List[RetrievalResult]:
        """内置航天知识回退——当所有外部检索源均无结果时使用。

        基于关键词匹配从内置知识库中检索相关条目。
        """
        query_lower = query.lower()
        scored: List[Tuple[float, str, str]] = []
        for text, category in self._BUILTIN_KNOWLEDGE:
            # 简单关键词匹配
            score = 0.0
            for word in query_lower.split():
                if len(word) > 1 and word in text.lower():
                    score += 1.0
            # 中文关键词匹配 (仅对 CJK 字符计分, 避免逐字符评分膨胀)
            for kw in query:
                if '\u4e00' <= kw <= '\u9fff' and kw in text:
                    score += 0.5
            if score > 0:
                scored.append((score, text, category))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, text, category in scored[:top_k]:
            results.append(RetrievalResult(
                text=text,
                score=min(score / 5.0, 1.0),  # 归一化到 [0, 1]
                source="builtin_knowledge",
                metadata={"category": category, "fallback": True},
                explanation="内置知识库回退",
            ))
        return results
