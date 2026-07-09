"""EvidenceVerifier — 证据验证器。

第一性原理（K5）：
    Agent 的回答必须有证据支撑。EvidenceVerifier 检查每个声明（claim）
    是否有检索到的证据（source）支持，给出支撑度评级和溯源链。

验证维度：
    1. 直接支撑：claim 中的关键事实在 source 中直接出现
    2. 间接支撑：claim 的语义与 source 高度相关
    3. 数值一致性：claim 中的数值与 source 中的数值匹配
    4. 矛盾检测：claim 与 source 是否矛盾
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .retriever import RetrievalResult


@dataclass
class EvidenceLink:
    """单条证据链接。

    Attributes:
        claim: 声明文本
        source_text: 支撑来源文本
        source_id: 来源 ID（文件名/节点 ID/URL）
        support_level: 支撑度（strong/weak/none/contradicts）
        matched_keywords: 匹配的关键词
        matched_numbers: 匹配的数值
        explanation: 说明
    """
    claim: str = ""
    source_text: str = ""
    source_id: str = ""
    support_level: str = "none"
    matched_keywords: List[str] = field(default_factory=list)
    matched_numbers: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class VerificationReport:
    """验证报告。"""
    overall_support: str = "none"  # strong / partial / weak / none / contradicts
    links: List[EvidenceLink] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_support": self.overall_support,
            "links": [l.to_dict() for l in self.links],
            "unsupported_claims": self.unsupported_claims,
            "confidence": self.confidence,
            "summary": self.summary,
        }


class EvidenceVerifier:
    """证据验证器——检查答案声明是否有检索证据支撑。

    用法：
        verifier = EvidenceVerifier()
        report = verifier.verify(
            answer="TLI 速度增量约 3.1 km/s，转移时间约 5 天",
            evidence=retrieval_results,
        )
        if report.overall_support == "strong":
            print("答案有充分证据支撑")
    """

    # 关键词匹配的最低重合度（Jaccard 系数）
    KEYWORD_THRESHOLD = 0.15
    # 数值匹配的相对容差
    NUMBER_TOLERANCE = 0.05

    def verify(self, answer: str,
               evidence: List[RetrievalResult]) -> VerificationReport:
        """验证答案是否有证据支撑。

        Args:
            answer: LLM 生成的答案文本
            evidence: 检索到的证据列表

        Returns:
            VerificationReport
        """
        if not evidence:
            return VerificationReport(
                overall_support="none", confidence=0.0,
                summary="无检索证据，无法验证",
            )

        # 1. 将答案切分为声明（按句号/换行/编号）
        claims = self._split_claims(answer)
        # 2. 为每个声明找最佳证据
        links: List[EvidenceLink] = []
        unsupported: List[str] = []
        strong_count = 0

        for claim in claims:
            best_link = self._find_best_evidence(claim, evidence)
            links.append(best_link)
            if best_link.support_level == "strong":
                strong_count += 1
            elif best_link.support_level == "none":
                unsupported.append(claim)

        # 3. 综合评级
        total = len(claims) or 1
        support_ratio = strong_count / total
        if support_ratio >= 0.7:
            overall = "strong"
        elif support_ratio >= 0.4:
            overall = "partial"
        elif support_ratio >= 0.2:
            overall = "weak"
        elif unsupported and not strong_count:
            overall = "none"
        else:
            overall = "partial"

        confidence = round(support_ratio, 3)

        return VerificationReport(
            overall_support=overall,
            links=links,
            unsupported_claims=unsupported,
            confidence=confidence,
            summary=f"{strong_count}/{total} 个声明有强证据支撑，"
                    f"{len(unsupported)} 个无支撑",
        )

    def _split_claims(self, text: str) -> List[str]:
        """将答案切分为声明列表。"""
        # 按句号、换行、编号切分 (仅匹配行首编号列表项, 不影响小数数值)
        parts = re.split(r'[。\n]|(?:(?<=^)|(?<=\n))\s*\d+\.\s+', text)
        claims = [p.strip() for p in parts if len(p.strip()) > 5]
        return claims or [text.strip()]

    def _find_best_evidence(self, claim: str,
                            evidence: List[RetrievalResult]) -> EvidenceLink:
        """为单个声明找最佳证据。"""
        best = EvidenceLink(claim=claim, support_level="none")
        claim_keywords = self._extract_keywords(claim)
        claim_numbers = self._extract_numbers(claim)

        for ev in evidence:
            ev_keywords = self._extract_keywords(ev.text)
            ev_numbers = self._extract_numbers(ev.text)

            # 关键词重合度
            matched_kw = list(claim_keywords & ev_keywords)
            jaccard = (len(matched_kw) /
                       len(claim_keywords | ev_keywords)) if (claim_keywords | ev_keywords) else 0

            # 数值匹配
            matched_nums = []
            for cn in claim_numbers:
                for en in ev_numbers:
                    if self._numbers_match(cn, en):
                        matched_nums.append(cn)
                        break

            # 矛盾检测（简单：claim 和 evidence 有相同关键词但数值矛盾）
            contradicts = False
            if matched_kw and claim_numbers and ev_numbers:
                if not matched_nums and jaccard > 0.2:
                    contradicts = True

            # 评级
            if contradicts:
                level = "contradicts"
            elif jaccard >= self.KEYWORD_THRESHOLD and matched_nums:
                level = "strong"
            elif jaccard >= self.KEYWORD_THRESHOLD:
                level = "weak"
            elif matched_nums:
                level = "weak"
            else:
                level = "none"

            # 评级优先级: strong > contradicts > weak > none
            _rank = {"strong": 3, "contradicts": 2, "weak": 1, "none": 0}
            if _rank.get(level, 0) > _rank.get(best.support_level, 0):
                best = EvidenceLink(
                    claim=claim,
                    source_text=ev.text[:200],
                    source_id=ev.metadata.get("doc_id",
                                ev.metadata.get("node_id", "unknown")),
                    support_level=level,
                    matched_keywords=matched_kw[:5],
                    matched_numbers=matched_nums,
                    explanation=f"关键词重合度={jaccard:.2f}, 数值匹配={len(matched_nums)}",
                )
                if level == "strong":
                    break

        return best

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """提取关键词（简单分词去停用词）。"""
        stop = {"的", "是", "在", "和", "与", "了", "为", "以", "及",
                "the", "a", "an", "is", "are", "in", "on", "at", "to", "for"}
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}|\d+\.?\d*', text.lower())
        return {w for w in words if w not in stop and len(w) > 1}

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """提取数值。"""
        return re.findall(r'\d+\.?\d*', text)

    def _numbers_match(self, a: str, b: str,
                       tolerance: float = None) -> bool:
        """数值是否匹配（相对容差内）。"""
        try:
            fa, fb = float(a), float(b)
            if fa == 0 and fb == 0:
                return True
            tol = tolerance or self.NUMBER_TOLERANCE
            return abs(fa - fb) / max(abs(fa), abs(fb), 1e-10) < tol
        except ValueError:
            return False
