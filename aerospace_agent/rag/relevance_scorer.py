"""摘要相关性评分模块 (Relevance Scorer)。

利用 LLM 对论文摘要与研究主题的相关性进行评估, 返回结构化评分。
当 LLM 不可用 (MockLLM 模式) 时, 回退到基于关键词重叠的规则评分。

* :class:`RelevanceScore`  — 相关性评分数据类
* :class:`RelevanceScorer`  — 相关性评分器 (LLM + 规则回退)

评分策略
--------
1. **LLM 模式**: 向 LLM 发送结构化 prompt, 要求返回 JSON:
   ``{"relevance": "strong"|"weak", "score": 0.0-1.0, "reason": "...", "key_concepts": [...]}``
2. **MockLLM 回退**: 基于关键词重叠 (TF-IDF 简化版):
   - 对摘要与研究主题分词, 计算研究主题词在摘要中的覆盖率
   - 覆盖率 > 0.3 -> ``strong``, 否则 ``weak``
   - 同时提取重叠的关键概念
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = ["RelevanceScore", "RelevanceScorer"]

# 航天领域常用停用词 (中英文), 评分时不计入
_STOP_WORDS = frozenset({
    # 英文停用词
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "be", "been", "by", "at", "from", "as",
    "this", "that", "these", "those", "it", "its", "we", "our", "their",
    "they", "them", "which", "who", "whom", "whose", "what", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    # 中文停用词 (单字)
    "的", "了", "在", "是", "和", "与", "或", "对", "为", "以",
    "及", "等", "由", "从", "向", "上", "下", "中", "其", "之",
    "一", "二", "三", "个", "也", "都", "还", "又", "就", "已",
    "可", "能", "会", "要", "该", "此", "那", "这", "些", "种",
})


# ---------------------------------------------------------------------------
# RelevanceScore 数据类
# ---------------------------------------------------------------------------
@dataclass
class RelevanceScore:
    """相关性评分数据类。

    Attributes:
        relevance:     相关性等级, ``"strong"`` (强相关) 或 ``"weak"`` (弱相关)
        score:         相关性分数, 0.0~1.0
        reason:        评分原因 (简要说明)
        key_concepts:  论文涉及的关键概念列表
    """

    relevance: str = "weak"
    score: float = 0.0
    reason: str = ""
    key_concepts: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        concepts = ", ".join(self.key_concepts[:5])
        return (
            f"RelevanceScore(relevance={self.relevance!r}, "
            f"score={self.score:.2f}, concepts=[{concepts}])"
        )

    @property
    def is_strong(self) -> bool:
        """是否强相关。"""
        return self.relevance == "strong"


# ---------------------------------------------------------------------------
# RelevanceScorer 评分器
# ---------------------------------------------------------------------------
class RelevanceScorer:
    """论文摘要相关性评分器。

    优先使用 LLM 进行语义级评估; 当 LLM 为 MockLLM 或调用失败时,
    回退到基于关键词重叠的规则评分。

    用法::

        scorer = RelevanceScorer()
        score = scorer.score_abstract(abstract, "地月转移轨道设计")
        if score.is_strong:
            print("强相关论文")
    """

    # MockLLM 回退的关键词重叠阈值: 覆盖率 > 此值判定为 strong
    OVERLAP_THRESHOLD = 0.3

    # 批量评分间隔 (秒)
    BATCH_INTERVAL = 1.0

    def __init__(self, llm_interface=None):
        """初始化评分器。

        Args:
            llm_interface: LLM 接口实例; 为 None 时通过 ``create_llm()`` 创建
        """
        if llm_interface is not None:
            self.llm = llm_interface
        else:
            self.llm = self._create_default_llm()

        # 判断是否为 MockLLM (按类名检测, 避免硬依赖)
        self._is_mock = self._check_is_mock(self.llm)

    # ------------------------------------------------------------------ 评分
    def score_abstract(self, abstract: str,
                       research_topic: str) -> RelevanceScore:
        """评估论文摘要与研究主题的相关性。

        Args:
            abstract:       论文摘要文本
            research_topic: 研究主题描述

        Returns:
            :class:`RelevanceScore` 评分结果
        """
        if not abstract or not abstract.strip():
            return RelevanceScore(
                relevance="weak", score=0.0,
                reason="摘要为空", key_concepts=[],
            )

        # MockLLM 模式: 直接用关键词重叠规则
        if self._is_mock:
            return self._rule_based_score(abstract, research_topic)

        # 真实 LLM 模式: 调用 LLM 评分
        try:
            return self._llm_score(abstract, research_topic)
        except Exception as e:
            print(f"[RelevanceScorer] LLM 评分失败, 回退到规则评分: {e}")
            return self._rule_based_score(abstract, research_topic)

    # ------------------------------------------------------------------ 批量评分
    def batch_score(
        self, papers: list, research_topic: str
    ) -> List[Tuple]:
        """批量评分多篇论文。

        每篇间隔 ``BATCH_INTERVAL`` 秒, 避免 LLM API 限速。

        Args:
            papers:          Paper 对象列表
            research_topic:  研究主题描述

        Returns:
            ``[(Paper, RelevanceScore), ...]`` 列表
        """
        results: List[Tuple] = []
        for i, paper in enumerate(papers):
            if i > 0:
                time.sleep(self.BATCH_INTERVAL)
            score = self.score_abstract(paper.abstract, research_topic)
            results.append((paper, score))
            print(f"  [{i+1}/{len(papers)}] {paper.id} -> "
                  f"{score.relevance} ({score.score:.2f})")
        return results

    # ------------------------------------------------------------------ LLM 评分
    def _llm_score(self, abstract: str,
                   research_topic: str) -> RelevanceScore:
        """调用 LLM 进行相关性评分, 解析 JSON 响应。"""
        prompt = self._build_prompt(abstract, research_topic)
        messages = [
            {"role": "system", "content": "你是航天轨道力学领域的文献评审专家。"},
            {"role": "user", "content": prompt},
        ]
        response = self.llm.chat(messages, temperature=0.3, max_tokens=500)
        return self._parse_llm_response(response, abstract, research_topic)

    @staticmethod
    def _build_prompt(abstract: str, research_topic: str) -> str:
        """构建 LLM 评分 prompt。"""
        return (
            f"你是航天轨道力学领域的文献评审专家。\n"
            f"研究主题：{research_topic}\n"
            f"论文摘要：{abstract}\n\n"
            f"请评估该论文与研究主题的相关性，返回 JSON：\n"
            f'{{"relevance": "strong"|"weak", '
            f'"score": 0.0-1.0, "reason": "简要原因", '
            f'"key_concepts": ["概念1", "概念2"]}}\n\n'
            f"只返回 JSON, 不要其他内容。"
        )

    @staticmethod
    def _parse_llm_response(response: str, abstract: str,
                            research_topic: str) -> RelevanceScore:
        """解析 LLM 返回的 JSON 响应。

        若解析失败, 回退到规则评分。
        """
        # 尝试从响应中提取 JSON 对象
        # LLM 可能把 JSON 包在 ```json ... ``` 代码块中
        json_str = response.strip()

        # 去除 markdown 代码块标记
        json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
        json_str = re.sub(r"\s*```$", "", json_str)

        # 尝试直接解析
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试从文本中提取第一个 {...} 块
            match = re.search(r"\{[^{}]*\}", json_str, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data is None:
            # JSON 解析失败, 回退到规则评分
            print("[RelevanceScorer] LLM 响应非 JSON, 回退到规则评分")
            return RelevanceScorer._rule_based_score_static(
                abstract, research_topic
            )

        # 提取字段
        relevance = str(data.get("relevance", "weak")).lower().strip()
        if relevance not in ("strong", "weak"):
            relevance = "weak"

        try:
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.8 if relevance == "strong" else 0.3

        reason = str(data.get("reason", "")).strip()
        key_concepts = data.get("key_concepts", [])
        if not isinstance(key_concepts, list):
            key_concepts = [str(key_concepts)]
        key_concepts = [str(c).strip() for c in key_concepts if str(c).strip()]

        return RelevanceScore(
            relevance=relevance,
            score=score,
            reason=reason,
            key_concepts=key_concepts,
        )

    # ------------------------------------------------------------------ 规则评分
    @staticmethod
    def _rule_based_score_static(abstract: str,
                                 research_topic: str) -> RelevanceScore:
        """基于关键词重叠的规则评分 (静态方法, 供回退使用)。

        简化版 TF-IDF 思路:
        1. 对研究主题和摘要分别分词 (中英文混合)
        2. 过滤停用词
        3. 计算研究主题词在摘要中的覆盖率
        4. 覆盖率 > 0.3 -> strong, 否则 weak
        """
        topic_words = RelevanceScorer._tokenize(research_topic)
        abstract_words = RelevanceScorer._tokenize(abstract)

        if not topic_words:
            return RelevanceScore(
                relevance="weak", score=0.0,
                reason="研究主题无有效关键词", key_concepts=[],
            )

        topic_set = set(topic_words)
        abstract_set = set(abstract_words)

        # 覆盖率: 研究主题词在摘要中出现的比例
        matched = topic_set & abstract_set
        coverage = len(matched) / len(topic_set)

        relevance = "strong" if coverage > RelevanceScorer.OVERLAP_THRESHOLD else "weak"

        # 提取关键概念: 交集词 + 摘要中的高频航天术语
        key_concepts = sorted(matched, key=lambda w: len(w), reverse=True)[:8]

        # 补充摘要中的航天领域术语
        aerospace_terms = RelevanceScorer._extract_aerospace_terms(abstract)
        for term in aerospace_terms:
            if term not in key_concepts:
                key_concepts.append(term)

        reason = (
            f"关键词覆盖率 {coverage:.1%} ({len(matched)}/{len(topic_set)} 个主题词命中)"
        )

        return RelevanceScore(
            relevance=relevance,
            score=round(coverage, 3),
            reason=reason,
            key_concepts=key_concepts[:10],
        )

    def _rule_based_score(self, abstract: str,
                          research_topic: str) -> RelevanceScore:
        """基于关键词重叠的规则评分 (实例方法)。"""
        return self._rule_based_score_static(abstract, research_topic)

    # ------------------------------------------------------------------ 分词
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中英文混合分词。

        * 英文: 连续字母数字作为一个词, 转小写
        * 中文: 逐字切分 (2-gram 也保留)
        * 过滤停用词
        """
        if not text:
            return []

        tokens: List[str] = []
        # 英文词
        for m in re.finditer(r"[a-zA-Z][a-zA-Z0-9\-]*", text):
            word = m.group().lower()
            if word not in _STOP_WORDS and len(word) >= 2:
                tokens.append(word)

        # 中文: 2-gram
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        for i in range(len(chinese_chars)):
            ch = chinese_chars[i]
            if ch not in _STOP_WORDS:
                tokens.append(ch)
            if i + 1 < len(chinese_chars):
                bigram = chinese_chars[i] + chinese_chars[i + 1]
                if bigram not in _STOP_WORDS:
                    tokens.append(bigram)

        return tokens

    # ------------------------------------------------------------------ 航天术语提取
    # 常见航天轨道力学术语 (用于 key_concepts 补充)
    _AEROSPACE_TERMS = [
        # 英文
        "orbit", "orbital", "transfer", "trajectory", "lunar", "moon",
        "earth", "spacecraft", "mission", "delta-v", "delta_v",
        "hohmann", "lambert", "kepler", "vis-viva", "patched", "conic",
        "soi", "tlI", "loi", "rendezvous", " docking", "attitude",
        "navigation", "propulsion", "thrust", "impulse", "maneuver",
        "ellipse", "hyperbolic", "parabolic", "eccentricity", "inclination",
        "ascending", "node", "perigee", "apogee", "apsis", "anomaly",
        "perturbation", "gravity", "celestial", "ephemeris",
        # 中文
        "轨道", "转移", "月球", "地月", "航天器", "任务", "变轨",
        "机动", "霍曼", "兰伯特", "开普勒", "活力", "拼凑", "圆锥",
        "作用球", "注入", "交会", "对接", "姿态", "导航", "推进",
        "脉冲", "椭圆", "双曲", "抛物", "偏心率", "倾角", "升交点",
        "近地点", "远地点", "摄动", "引力", "天体", "历书",
    ]

    @classmethod
    def _extract_aerospace_terms(cls, text: str) -> List[str]:
        """从文本中提取航天领域术语。"""
        low = text.lower()
        found = []
        for term in cls._AEROSPACE_TERMS:
            tl = term.lower().strip()
            if tl and tl in low:
                found.append(term.strip())
        return found

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _create_default_llm():
        """通过 create_llm() 创建默认 LLM。"""
        import sys
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        from core.llm_interface import create_llm
        return create_llm()

    @staticmethod
    def _check_is_mock(llm) -> bool:
        """检测 LLM 是否为 MockLLM (按类名, 避免硬依赖)。"""
        return type(llm).__name__ == "MockLLM"


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("RelevanceScorer 自测")
    print("=" * 70)

    scorer = RelevanceScorer()
    print(f"LLM 类型: {type(scorer.llm).__name__}")
    print(f"MockLLM 模式: {scorer._is_mock}")

    # 测试用例: 论文摘要 + 研究主题
    test_cases = [
        {
            "topic": "地月转移轨道设计 (lunar transfer orbit design)",
            "abstract": (
                "This paper presents a low-thrust trajectory design method "
                "for lunar transfer orbits using differential dynamic "
                "programming. The spacecraft is guided from Earth orbit to "
                "a lunar transfer orbit with feedback control. "
                "Monte Carlo simulations confirm successful guidance to the "
                "lunar transfer orbit under operational uncertainties."
            ),
        },
        {
            "topic": "地月转移轨道设计",
            "abstract": (
                "We study the formation and evolution of galaxies in the "
                "early universe using data from the James Webb Space "
                "Telescope. Spectroscopic observations reveal the star "
                "formation history of distant galaxies at redshift z > 10."
            ),
        },
        {
            "topic": "lunar transfer orbit",
            "abstract": (
                "本文研究了地月转移轨道的优化设计问题, 采用拼凑圆锥近似方法, "
                "结合 Hohmann 转移和 Lambert 问题求解, 计算了 TLI 和 LOI 的 "
                "速度增量需求, 并分析了发射窗口与相位角的约束关系。"
            ),
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i} ---")
        print(f"研究主题: {tc['topic']}")
        print(f"摘要前80字: {tc['abstract'][:80]}...")
        score = scorer.score_abstract(tc["abstract"], tc["topic"])
        print(f"评分结果: {score}")
        print(f"  相关性: {score.relevance}")
        print(f"  分数:   {score.score:.3f}")
        print(f"  原因:   {score.reason}")
        print(f"  概念:   {score.key_concepts}")

    # 批量评分测试 (用模拟 Paper)
    print("\n--- 批量评分测试 ---")
    from literature_fetcher import Paper

    mock_papers = [
        Paper(
            id="2401.00001",
            title="Lunar Transfer Orbit Design via Hohmann",
            authors=["Test Author"],
            abstract=(
                "This paper discusses lunar transfer orbit design using "
                "Hohmann transfer and vis-viva equation for delta-v "
                "calculation. The patched conic approximation is applied."
            ),
        ),
        Paper(
            id="2401.00002",
            title="Galaxy Formation in Early Universe",
            authors=["Another Author"],
            abstract=(
                "We study galaxy formation using JWST data at high redshift, "
                "focusing on stellar mass assembly and dark matter halos."
            ),
        ),
    ]

    results = scorer.batch_score(mock_papers, "lunar transfer orbit")
    print(f"\n批量评分完成: {len(results)} 篇")
    for paper, score in results:
        print(f"  {paper.id}: {score.relevance} ({score.score:.2f}) - {paper.title[:50]}")
