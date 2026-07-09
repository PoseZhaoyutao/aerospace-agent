"""文献处理管线 (Literature Pipeline) —— 核心编排模块。

将文献搜索、相关性评分、PDF 下载、全文总结、RAG 索引、知识图谱更新
串联为完整的自动化管线, 让 Agent 能自动获取最新航天文献并纳入知识库。

管线流程::

    1. 登录 CSTCloud (或回退到默认 LLM)
    2. 搜索 arXiv 文献
    3. 逐篇评分 (LLM 或关键词重叠规则)
    4. strong 相关 -> 下载 PDF + 全文总结 + 索引 + 图谱更新
    5. weak 相关   -> 跳过, 仅记录元信息
    6. 生成管线报告
    7. 触发知识云图更新

主要类
------
* :class:`PaperRecord`    — 单篇论文处理记录
* :class:`PipelineReport` — 管线运行报告
* :class:`LiteraturePipeline` — 管线编排器
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 包内相对导入——不再使用 sys.path hack
try:
    from .literature_fetcher import (
        ArxivFetcher,
        CSTCloudAuthenticator,
        Paper,
        extract_text_from_pdf,
        DEFAULT_PAPER_DIR,
    )
    from .relevance_scorer import RelevanceScore, RelevanceScorer
except ImportError:
    # 直接运行脚本时的回退导入 (使用完整包路径)
    from aerospace_agent.rag.literature_fetcher import (
        ArxivFetcher,
        CSTCloudAuthenticator,
        Paper,
        extract_text_from_pdf,
        DEFAULT_PAPER_DIR,
    )
    from aerospace_agent.rag.relevance_scorer import RelevanceScore, RelevanceScorer

__all__ = [
    "PaperRecord",
    "PipelineReport",
    "LiteraturePipeline",
]

# 知识图谱节点/边类型常量
_NODE_PAPER = "paper"
_NODE_CONCEPT = "concept"
_REL_USED_BY = "used_by"


# ---------------------------------------------------------------------------
# PaperRecord 数据类
# ---------------------------------------------------------------------------
@dataclass
class PaperRecord:
    """单篇论文的处理记录。

    Attributes:
        paper:     原始 Paper 对象
        score:     相关性评分
        status:    处理状态: ``"downloaded"`` (已下载处理) |
                   ``"skipped"`` (弱相关跳过) | ``"failed"`` (处理失败)
        summary:   全文总结 (仅 strong 相关论文有值)
        pdf_path:  PDF 本地路径 (仅已下载的有值)
        concepts:  论文涉及的关键概念列表
        indexed:   是否已索引到 RAG
    """

    paper: Paper
    score: RelevanceScore
    status: str = "skipped"
    summary: Optional[str] = None
    pdf_path: Optional[str] = None
    concepts: List[str] = field(default_factory=list)
    indexed: bool = False

    def __repr__(self) -> str:
        return (
            f"PaperRecord(id={self.paper.id!r}, status={self.status!r}, "
            f"relevance={self.score.relevance!r}, indexed={self.indexed})"
        )


# ---------------------------------------------------------------------------
# PipelineReport 数据类
# ---------------------------------------------------------------------------
@dataclass
class PipelineReport:
    """管线运行报告。

    Attributes:
        research_topic:          研究主题
        total_found:             搜索到的文献总数
        strong_count:            强相关论文数
        weak_count:              弱相关论文数
        downloaded_count:        成功下载并处理的论文数
        papers:                  所有论文的处理记录列表
        knowledge_graph_snapshot: 知识图谱快照 (节点数、边数等)
    """

    research_topic: str = ""
    total_found: int = 0
    strong_count: int = 0
    weak_count: int = 0
    downloaded_count: int = 0
    papers: List[PaperRecord] = field(default_factory=list)
    knowledge_graph_snapshot: dict = field(default_factory=dict)

    def summary_text(self) -> str:
        """生成可读的报告摘要文本。"""
        lines = [
            "=" * 70,
            "文献处理管线报告",
            "=" * 70,
            f"研究主题:       {self.research_topic}",
            f"搜索到文献:     {self.total_found} 篇",
            f"强相关 (strong): {self.strong_count} 篇",
            f"弱相关 (weak):   {self.weak_count} 篇",
            f"已下载处理:     {self.downloaded_count} 篇",
            "-" * 70,
        ]
        # 知识图谱快照
        snap = self.knowledge_graph_snapshot
        if snap:
            lines.append(f"知识图谱节点数: {snap.get('num_nodes', '?')}")
            lines.append(f"知识图谱边数:   {snap.get('num_edges', '?')}")
            new_nodes = snap.get("new_nodes", [])
            if new_nodes:
                lines.append(f"新增节点:       {len(new_nodes)} 个")
                for nid in new_nodes[:10]:
                    lines.append(f"  + {nid}")
            lines.append("-" * 70)
        # 论文详情
        for i, rec in enumerate(self.papers, 1):
            tag = rec.status.upper()
            lines.append(
                f"  {i}. [{tag}] {rec.paper.id} "
                f"({rec.score.relevance}, {rec.score.score:.2f}) "
                f"{rec.paper.title[:50]}"
            )
            if rec.summary:
                lines.append(f"     总结: {rec.summary[:100]}...")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LiteraturePipeline 管线编排器
# ---------------------------------------------------------------------------
class LiteraturePipeline:
    """文献处理管线编排器。

    组合 RAG + ArxivFetcher + RelevanceScorer, 实现从搜索到索引的全流程自动化。

    用法::

        pipeline = LiteraturePipeline()
        report = pipeline.run("lunar transfer orbit", max_papers=5)
        print(report.summary_text())
    """

    def __init__(
        self,
        rag=None,
        fetcher: Optional[ArxivFetcher] = None,
        scorer: Optional[RelevanceScorer] = None,
    ):
        """初始化管线。

        Args:
            rag:     AerospaceRAG 实例; 为 None 时自动创建
            fetcher: ArxivFetcher 实例; 为 None 时自动创建
            scorer:  RelevanceScorer 实例; 为 None 时自动创建
        """
        # RAG 系统
        if rag is not None:
            self.rag = rag
        else:
            self.rag = self._create_default_rag()

        # 文献搜索器
        self.fetcher = fetcher or ArxivFetcher()

        # 相关性评分器
        self.scorer = scorer or RelevanceScorer()

        # CSTCloud 认证器
        self.authenticator = CSTCloudAuthenticator()

        # LLM 接口 (登录后设置)
        self.llm = None

        # 记录知识图谱初始状态 (用于计算变化)
        self._initial_kg_nodes = 0
        self._initial_kg_edges = 0
        self._new_nodes: List[str] = []

    # ------------------------------------------------------------------ 搜索+评分
    def search_and_evaluate(
        self,
        query: str,
        research_topic: str,
        max_results: int = 10,
    ) -> List[Tuple[Paper, RelevanceScore]]:
        """搜索 arXiv 文献并逐篇评分。

        Args:
            query:          arXiv 搜索关键词
            research_topic: 相关性评分的研究主题
            max_results:    最大搜索条数

        Returns:
            ``[(Paper, RelevanceScore), ...]`` 列表
        """
        print(f"\n[Pipeline] 搜索 arXiv: query={query!r}, max={max_results}")
        papers = self.fetcher.search(query, max_results=max_results)

        if not papers:
            print("[Pipeline] 未找到文献")
            return []

        print(f"[Pipeline] 对 {len(papers)} 篇文献进行相关性评分 "
              f"(主题: {research_topic})")
        results = self.scorer.batch_score(papers, research_topic)
        return results

    # ------------------------------------------------------------------ 处理 strong 论文
    def process_strong_paper(
        self,
        paper: Paper,
        score: RelevanceScore,
        research_topic: str,
    ) -> PaperRecord:
        """处理强相关论文: 下载 PDF + 提取全文 + 生成总结 + 索引 + 图谱更新。

        Args:
            paper:          论文对象
            score:          相关性评分
            research_topic: 研究主题 (用于总结的关联分析)

        Returns:
            :class:`PaperRecord` 处理记录
        """
        print(f"\n[Pipeline] === 处理 strong 论文: {paper.id} ===")
        print(f"  标题: {paper.title}")
        print(f"  作者: {', '.join(paper.authors[:3])}")

        record = PaperRecord(
            paper=paper,
            score=score,
            status="failed",
            concepts=list(score.key_concepts),
        )

        # 步骤 1: 下载 PDF
        print(f"  [1/5] 下载 PDF...")
        pdf_path = self.fetcher.download_pdf(paper)
        record.pdf_path = pdf_path

        if not pdf_path:
            print(f"  PDF 下载失败, 使用摘要作为替代")
            # 下载失败仍可继续: 用摘要生成总结
            full_text = paper.abstract
        else:
            record.status = "downloaded"
            # 步骤 2: 提取全文
            print(f"  [2/5] 提取全文文本...")
            full_text = extract_text_from_pdf(pdf_path)
            # 如果提取的文本太短, 用摘要补充
            if len(full_text.strip()) < 200:
                print(f"  提取文本过短 ({len(full_text)} 字符), 用摘要补充")
                full_text = paper.abstract + "\n\n" + full_text

        # 步骤 3: 生成全文总结
        print(f"  [3/5] 生成全文总结...")
        summary = self._generate_summary(paper, full_text, research_topic)
        record.summary = summary
        print(f"  总结长度: {len(summary)} 字符")

        # 步骤 4: 索引到 RAG
        print(f"  [4/5] 索引到 RAG...")
        self._index_to_rag(paper, summary, score)
        record.indexed = True

        # 步骤 5: 更新知识图谱
        print(f"  [5/5] 更新知识图谱...")
        self._update_knowledge_graph(paper, score)
        print(f"  完成: {paper.id}")

        return record

    # ------------------------------------------------------------------ 处理 weak 论文
    def process_weak_paper(
        self,
        paper: Paper,
        score: RelevanceScore,
    ) -> PaperRecord:
        """处理弱相关论文: 仅记录元信息, 不下载全文。

        Args:
            paper: 论文对象
            score: 相关性评分

        Returns:
            :class:`PaperRecord` 处理记录
        """
        print(f"\n[Pipeline] --- 跳过 weak 论文: {paper.id} ---")
        print(f"  标题: {paper.title}")
        print(f"  原因: {score.reason}")

        return PaperRecord(
            paper=paper,
            score=score,
            status="skipped",
            summary=None,
            pdf_path=None,
            concepts=list(score.key_concepts),
            indexed=False,
        )

    # ------------------------------------------------------------------ 完整管线
    def run(
        self,
        research_topic: str,
        max_papers: int = 10,
        min_relevance: str = "strong",
    ) -> PipelineReport:
        """运行完整文献处理管线。

        Args:
            research_topic: 研究主题 (同时作为搜索关键词和评分主题)
            max_papers:     最大处理论文数
            min_relevance:  最低相关性阈值, ``"strong"`` 或 ``"weak"``;
                            仅处理达到此级别的论文

        Returns:
            :class:`PipelineReport` 管线报告
        """
        print("=" * 70)
        print(f"[Pipeline] 启动文献处理管线")
        print(f"  研究主题:   {research_topic}")
        print(f"  最大论文数: {max_papers}")
        print(f"  最低相关性: {min_relevance}")
        print("=" * 70)

        # 步骤 1: 登录 CSTCloud (或回退到默认 LLM)
        print("\n[Pipeline] 步骤 1: 登录 CSTCloud (或回退到默认 LLM)...")
        self.authenticator.login()
        self.llm = self.authenticator.get_llm()
        llm_type = type(self.llm).__name__
        if self.authenticator.is_authenticated:
            print(f"  CSTCloud 认证成功, LLM: {llm_type}")
        else:
            print(f"  回退到默认 LLM: {llm_type}")
        # 更新 scorer 的 LLM (使用管线级 LLM)
        self.scorer.llm = self.llm
        self.scorer._is_mock = self.scorer._check_is_mock(self.llm)

        # 记录知识图谱初始状态
        kg = self._get_knowledge_graph()
        if kg:
            self._initial_kg_nodes = kg.num_nodes
            self._initial_kg_edges = kg.num_edges
            print(f"  知识图谱初始: {self._initial_kg_nodes} 节点, "
                  f"{self._initial_kg_edges} 边")

        # 步骤 2: 搜索 arXiv 文献
        print("\n[Pipeline] 步骤 2: 搜索 arXiv 文献...")
        scored = self.search_and_evaluate(
            research_topic, research_topic, max_results=max_papers
        )

        if not scored:
            print("[Pipeline] 未找到文献, 管线结束")
            return PipelineReport(
                research_topic=research_topic,
                total_found=0,
                knowledge_graph_snapshot=self._kg_snapshot(),
            )

        # 步骤 3: 逐篇评分 (已在 search_and_evaluate 中完成)
        print(f"\n[Pipeline] 步骤 3: 评分完成, 共 {len(scored)} 篇")

        # 统计
        strong_list = [(p, s) for p, s in scored if s.relevance == "strong"]
        weak_list = [(p, s) for p, s in scored if s.relevance == "weak"]

        # 步骤 4: 处理 strong 相关论文
        print(f"\n[Pipeline] 步骤 4: 处理 {len(strong_list)} 篇 strong 论文...")
        records: List[PaperRecord] = []
        downloaded = 0
        for i, (paper, score) in enumerate(strong_list):
            print(f"\n  ({i+1}/{len(strong_list)})")
            record = self.process_strong_paper(paper, score, research_topic)
            records.append(record)
            if record.status == "downloaded":
                downloaded += 1

        # 步骤 5: 处理 weak 相关论文 (仅记录)
        if min_relevance == "weak":
            print(f"\n[Pipeline] 步骤 5: 处理 {len(weak_list)} 篇 weak 论文...")
            for paper, score in weak_list:
                record = self.process_weak_paper(paper, score)
                records.append(record)
        else:
            # weak 论文仅简单记录
            for paper, score in weak_list:
                records.append(self.process_weak_paper(paper, score))

        # 步骤 6: 生成管线报告
        print(f"\n[Pipeline] 步骤 6: 生成管线报告...")
        report = PipelineReport(
            research_topic=research_topic,
            total_found=len(scored),
            strong_count=len(strong_list),
            weak_count=len(weak_list),
            downloaded_count=downloaded,
            papers=records,
            knowledge_graph_snapshot=self._kg_snapshot(),
        )

        # 步骤 7: 触发知识云图更新
        print(f"\n[Pipeline] 步骤 7: 触发知识云图更新...")
        self._trigger_knowledge_cloud()

        # 保存 RAG
        try:
            self.rag.save()
            print("[Pipeline] RAG 已保存")
        except Exception as e:
            print(f"[Pipeline] RAG 保存失败: {e}")

        # 统一重建向量索引 (避免每篇论文后重复 reindex 的性能开销)
        try:
            self.rag.kb.vector_store.reindex()
            print("[Pipeline] 向量索引已重建")
        except Exception as e:
            print(f"[Pipeline] 向量索引重建失败: {e}")

        return report

    # ------------------------------------------------------------------ 生成总结
    def _generate_summary(
        self, paper: Paper, full_text: str, research_topic: str
    ) -> str:
        """用 LLM 生成论文全文总结。

        结构化输出: 研究背景 | 方法 | 关键公式/算法 | 主要结论 | 与研究主题的关联。
        若 LLM 为 MockLLM 或调用失败, 回退到基于摘要的模板总结。
        """
        # 截取前 8000 字符 (避免 token 超限)
        text_snippet = full_text[:8000]

        prompt = (
            f"请总结这篇航天领域论文的核心内容，结构化输出：\n"
            f"研究背景 | 方法 | 关键公式/算法 | 主要结论 | "
            f"与\"{research_topic}\"的关联\n"
            f"限 500 字以内。\n\n"
            f"论文标题: {paper.title}\n"
            f"作者: {', '.join(paper.authors[:5])}\n"
            f"全文内容:\n{text_snippet}"
        )

        # 检查是否为 MockLLM
        is_mock = type(self.llm).__name__ == "MockLLM"

        if is_mock:
            return self._mock_summary(paper, full_text, research_topic)

        try:
            messages = [
                {"role": "system", "content": "你是航天领域的文献分析专家。"},
                {"role": "user", "content": prompt},
            ]
            response = self.llm.chat(messages, temperature=0.3, max_tokens=800)
            if response and len(response.strip()) > 20:
                return response.strip()
            # 响应过短, 回退
            return self._mock_summary(paper, full_text, research_topic)
        except Exception as e:
            print(f"  LLM 总结失败, 回退到模板: {e}")
            return self._mock_summary(paper, full_text, research_topic)

    @staticmethod
    def _mock_summary(
        paper: Paper, full_text: str, research_topic: str
    ) -> str:
        """MockLLM 模式下的模板总结 (基于摘要和提取文本)。"""
        abstract = paper.abstract or ""
        # 从全文中提取前几段作为方法描述
        text_lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        method_hint = " ".join(text_lines[1:4])[:300] if len(text_lines) > 1 else ""

        return (
            f"【论文总结】\n"
            f"标题: {paper.title}\n"
            f"arXiv ID: {paper.id}\n"
            f"作者: {', '.join(paper.authors[:5])}\n\n"
            f"研究背景: {abstract[:200]}\n\n"
            f"方法: {method_hint or '(详见全文)'}\n\n"
            f"关键公式/算法: (基于全文分析, 详见原文)\n\n"
            f"主要结论: {abstract[200:400] if len(abstract) > 200 else '(详见全文)'}\n\n"
            f"与「{research_topic}」的关联: 该论文涉及航天轨道力学相关主题, "
            f"可能为研究提供方法参考或对比基准。\n"
        )

    # ------------------------------------------------------------------ 索引到 RAG
    def _index_to_rag(
        self, paper: Paper, summary: str, score: RelevanceScore
    ) -> None:
        """将论文总结索引到 RAG 系统。

        metadata 包含 paper_id, title, authors, arxiv_id, source='arxiv'。
        """
        try:
            metadata = {
                "paper_id": paper.id,
                "title": paper.title,
                "authors": ", ".join(paper.authors[:5]),
                "arxiv_id": paper.id,
                "source": "arxiv",
                "type": "paper",
                "relevance": score.relevance,
                "relevance_score": score.score,
                "primary_category": paper.primary_category or "",
                "published_date": paper.published_date,
            }
            doc_id = self.rag.kb.index_text(
                summary, source="arxiv", metadata=metadata
            )
            print(f"  已索引到 RAG (doc_id={doc_id})")
        except Exception as e:
            print(f"  RAG 索引失败: {e}")

    # ------------------------------------------------------------------ 更新知识图谱
    def _update_knowledge_graph(
        self, paper: Paper, score: RelevanceScore
    ) -> None:
        """将论文的关键概念添加到知识图谱。

        - 创建 paper 类型节点
        - 连接论文节点与其涉及的已有概念节点
        - 如果概念是新的, 也创建概念节点
        """
        kg = self._get_knowledge_graph()
        if kg is None:
            print("  知识图谱不可用, 跳过")
            return

        # 创建论文节点
        paper_node_id = f"paper:{paper.id}"
        paper_content = (
            f"论文: {paper.title}\n"
            f"作者: {', '.join(paper.authors[:5])}\n"
            f"arXiv: {paper.id}\n"
            f"摘要: {paper.abstract[:200]}"
        )
        if not kg.has_node(paper_node_id):
            kg.add_node(
                paper_node_id,
                type=_NODE_PAPER,
                content=paper_content,
                metadata={
                    "arxiv_id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "source": "arxiv",
                    "primary_category": paper.primary_category or "",
                    "published_date": paper.published_date,
                    "aliases": [paper.id, paper.short_id],
                },
            )
            self._new_nodes.append(paper_node_id)
            print(f"  新增论文节点: {paper_node_id}")
        else:
            print(f"  论文节点已存在: {paper_node_id}")

        # 连接论文节点与概念节点
        connected = 0
        for concept in score.key_concepts:
            if not concept or len(concept) < 2:
                continue
            # 尝试在图谱中匹配已有概念
            matches = kg.match_concepts(concept)
            if matches:
                # 连接到最佳匹配的概念节点
                best_match = matches[0][0]
                try:
                    kg.add_edge(best_match, paper_node_id, _REL_USED_BY, 1.0)
                    connected += 1
                except KeyError:
                    pass
            else:
                # 概念是新的, 创建概念节点并连接
                concept_id = f"concept:{concept.lower().replace(' ', '_')}"
                if not kg.has_node(concept_id):
                    kg.add_node(
                        concept_id,
                        type=_NODE_CONCEPT,
                        content=f"概念: {concept} (来自论文 {paper.id})",
                        metadata={
                            "source": "arxiv",
                            "from_paper": paper.id,
                            "aliases": [concept],
                        },
                    )
                    self._new_nodes.append(concept_id)
                try:
                    kg.add_edge(concept_id, paper_node_id, _REL_USED_BY, 1.0)
                    connected += 1
                except KeyError:
                    pass

        print(f"  知识图谱: 新增 {len(self._new_nodes)} 节点, "
              f"连接 {connected} 条边")

    # ------------------------------------------------------------------ 知识图谱快照
    def _kg_snapshot(self) -> dict:
        """生成知识图谱快照。"""
        kg = self._get_knowledge_graph()
        if kg is None:
            return {}
        return {
            "num_nodes": kg.num_nodes,
            "num_edges": kg.num_edges,
            "initial_nodes": self._initial_kg_nodes,
            "initial_edges": self._initial_kg_edges,
            "new_nodes": self._new_nodes,
            "node_delta": kg.num_nodes - self._initial_kg_nodes,
            "edge_delta": kg.num_edges - self._initial_kg_edges,
        }

    # ------------------------------------------------------------------ 触发知识云图
    def _trigger_knowledge_cloud(self) -> None:
        """触发知识云图更新。

        调用 ``KnowledgeCloudGenerator.generate()`` 生成自包含 HTML 力导向云图;
        若模块不可用, 优雅降级为打印知识图谱摘要。
        """
        kg = self._get_knowledge_graph()
        if kg is None:
            print("[Pipeline] 知识图谱不可用, 跳过云图更新")
            return

        try:
            try:
                from .knowledge_cloud import KnowledgeCloudGenerator
            except ImportError:
                from aerospace_agent.rag.knowledge_cloud import KnowledgeCloudGenerator
            gen = KnowledgeCloudGenerator()
            output_path = gen.generate(
                kg,
                output_path=os.path.join(os.getcwd(), "reports", "knowledge_cloud.html"),
                title=f"航天知识云图 (文献管线更新)",
            )
            print(f"[Pipeline] 知识云图已更新: {output_path}")
            print(f"[Pipeline] 当前图谱: {kg.num_nodes} 节点, {kg.num_edges} 边")
        except ImportError:
            # 优雅降级: 打印知识图谱摘要
            print(f"[Pipeline] 知识云图模块不可用, 当前图谱状态: "
                  f"{kg.num_nodes} 节点, {kg.num_edges} 边")
        except Exception as e:
            print(f"[Pipeline] 知识云图更新异常: {e}")
            print(f"[Pipeline] 当前图谱: {kg.num_nodes} 节点, {kg.num_edges} 边")

    # ------------------------------------------------------------------ 工具方法
    def _get_knowledge_graph(self):
        """获取 RAG 系统的知识图谱实例。"""
        try:
            return self.rag.kb.knowledge_graph
        except AttributeError:
            return None

    @staticmethod
    def _create_default_rag():
        """创建默认 AerospaceRAG 实例。"""
        try:
            from .aerospace_rag import AerospaceRAG
        except ImportError:
            from aerospace_agent.rag.aerospace_rag import AerospaceRAG
        return AerospaceRAG()


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LiteraturePipeline 自测 (查询: lunar transfer orbit)")
    print("=" * 70)

    # 创建管线 (使用临时数据目录避免污染已有索引)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            from .aerospace_rag import AerospaceRAG
        except ImportError:
            from aerospace_agent.rag.aerospace_rag import AerospaceRAG

        rag = AerospaceRAG(data_dir=tmpdir, autoload=False)
        pipeline = LiteraturePipeline(rag=rag)

        # 运行完整管线
        report = pipeline.run(
            research_topic="lunar transfer orbit",
            max_papers=3,
            min_relevance="strong",
        )

        # 打印报告
        print()
        print(report.summary_text())

        # 验证: 检查知识图谱变化
        snap = report.knowledge_graph_snapshot
        if snap:
            print(f"\n[验证] 知识图谱变化:")
            print(f"  节点: {snap['initial_nodes']} -> {snap['num_nodes']} "
                  f"(+{snap['node_delta']})")
            print(f"  边:   {snap['initial_edges']} -> {snap['num_edges']} "
                  f"(+{snap['edge_delta']})")
            if snap["new_nodes"]:
                print(f"  新增节点: {snap['new_nodes']}")

        # 验证: 检查 RAG 检索
        if report.downloaded_count > 0:
            print(f"\n[验证] RAG 检索测试:")
            ctx = rag.query("lunar transfer orbit", top_k=3)
            print(ctx[:500])
