"""Traceability — 可追溯性工具。

第一性原理（K5）：
    Agent 的每个回答都必须可追溯到来源。Traceability 为答案附上
    source_id 链，使任何结论都能回溯到原始文档/数据/代码。

    答案不是"凭空生成"的，而是"从证据合成"的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .retriever import RetrievalResult
from .verifier import VerificationReport


@dataclass
class TraceEntry:
    """单条溯源记录。

    Attributes:
        source_id: 来源 ID
        source_type: 来源类型（document/code/memory/web/database）
        source_ref: 来源引用（文件路径/URL/节点 ID）
        excerpt: 摘录片段
        retrieval_score: 检索得分
        used_in_answer: 是否被答案使用
    """
    source_id: str = ""
    source_type: str = ""
    source_ref: str = ""
    excerpt: str = ""
    retrieval_score: float = 0.0
    used_in_answer: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AnswerTrace:
    """答案的完整溯源链。

    Attributes:
        answer: 答案文本
        query: 原始查询
        trace_entries: 溯源记录列表
        verification: 证据验证报告
        timestamp: 生成时间
        agent_version: Agent 版本
        retrieval_sources: 检索使用的源
    """
    answer: str = ""
    query: str = ""
    trace_entries: List[TraceEntry] = field(default_factory=list)
    verification: Optional[VerificationReport] = None
    timestamp: str = ""
    agent_version: str = "0.1.0"
    retrieval_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "query": self.query,
            "trace_entries": [t.to_dict() for t in self.trace_entries],
            "verification": self.verification.to_dict() if self.verification else None,
            "timestamp": self.timestamp,
            "agent_version": self.agent_version,
            "retrieval_sources": self.retrieval_sources,
        }

    def to_citation_string(self) -> str:
        """生成引用字符串（附在答案末尾）。"""
        if not self.trace_entries:
            return self.answer
        used = [t for t in self.trace_entries if t.used_in_answer]
        if not used:
            return self.answer
        refs = []
        for i, t in enumerate(used, 1):
            ref = t.source_ref or t.source_id
            refs.append(f"[{i}] {ref}")
        return f"{self.answer}\n\n---\n来源:\n" + "\n".join(refs)


class TraceabilityManager:
    """可追溯性管理器——为答案构建溯源链。

    用法：
        tracer = TraceabilityManager()
        trace = tracer.build_trace(
            answer="TLI 速度增量约 3.1 km/s",
            query="地月转移轨道 TLI 速度增量",
            evidence=retrieval_results,
            verification=verifier_report,
        )
        cited = trace.to_citation_string()  # 答案 + 来源引用
    """

    def build_trace(self, answer: str, query: str,
                    evidence: List[RetrievalResult],
                    verification: VerificationReport = None,
                    retrieval_sources: List[str] = None) -> AnswerTrace:
        """为答案构建完整溯源链。

        Args:
            answer: LLM 生成的答案
            query: 原始查询
            evidence: 检索到的证据
            verification: 证据验证报告
            retrieval_sources: 使用的检索源列表

        Returns:
            AnswerTrace
        """
        trace_entries: List[TraceEntry] = []

        # 确定哪些证据被答案使用（通过验证报告的 links）
        used_source_ids = set()
        if verification:
            for link in verification.links:
                if link.support_level in ("strong", "weak"):
                    used_source_ids.add(link.source_id)

        for ev in evidence:
            source_id = str(ev.metadata.get("doc_id",
                             ev.metadata.get("node_id", "unknown")))
            source_type = ev.metadata.get("source_type", "document")
            source_ref = ev.metadata.get("file_path",
                            ev.metadata.get("url",
                            ev.metadata.get("node_id", source_id)))
            trace_entries.append(TraceEntry(
                source_id=source_id,
                source_type=source_type,
                source_ref=source_ref,
                excerpt=ev.text[:150],
                retrieval_score=ev.score,
                used_in_answer=source_id in used_source_ids,
            ))

        return AnswerTrace(
            answer=answer,
            query=query,
            trace_entries=trace_entries,
            verification=verification,
            timestamp=datetime.now().isoformat(),
            retrieval_sources=retrieval_sources or [],
        )

    def format_trace_for_display(self, trace: AnswerTrace) -> str:
        """格式化溯源链用于展示。"""
        lines = [
            f"查询: {trace.query}",
            f"答案: {trace.answer[:200]}...",
            f"时间: {trace.timestamp}",
            f"检索源: {', '.join(trace.retrieval_sources) or 'N/A'}",
            "",
            "溯源链:",
        ]
        for i, t in enumerate(trace.trace_entries, 1):
            mark = "✓" if t.used_in_answer else " "
            lines.append(
                f"  {mark} [{i}] ({t.source_type}) {t.source_ref} "
                f"score={t.retrieval_score:.3f}"
            )
            lines.append(f"      摘录: {t.excerpt[:80]}...")
        if trace.verification:
            lines.append("")
            lines.append(
                f"验证: {trace.verification.overall_support} "
                f"(confidence={trace.verification.confidence:.2f})"
            )
        return "\n".join(lines)
