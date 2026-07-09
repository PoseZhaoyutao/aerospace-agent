"""航天文献综述工作流 (Literature Review Workflow)。

将「搜索文献 → 相关性评分 → PDF 下载/全文总结/RAG 索引/知识图谱更新 →
知识云图生成 → 知识学习报告生成」串联为一条完整的自动化工作流,
让 Agent 能按一句提示词 (如"帮我搜索文献、作报告、做云图")完成端到端的
文献调研与知识沉淀。

核心编排
--------
* :class:`LiteraturePipeline` —— 文献搜索/评分/下载/索引/图谱更新 (RAG 层)
* :class:`KnowledgeCloudGenerator` —— 交互式力导向知识云图 HTML
* :class:`KnowledgeReportGenerator` —— 自包含知识学习报告 HTML (内嵌云图)

LLM 注入
--------
本工作流支持通过 ``llm`` 参数注入任意 :class:`LLMInterface` 实例
(如 Qwen3 ``LocalLLM``)。注入后会用一个轻量适配器替换管线内部的
``CSTCloudAuthenticator``, 使搜索评分、全文总结等 LLM 调用全部走注入的
模型; 未注入时回退到管线的默认认证/回退逻辑 (CSTCloud → create_llm → MockLLM)。

输出
----
* 知识云图 HTML (reports/knowledge_cloud_<topic>.html)
* 知识学习报告 HTML (reports/literature_review_<topic>.html, 内嵌云图)
* PipelineReport 摘要文本
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from .base import BaseWorkflow, WorkflowResult, register_workflow

__all__ = ["LiteratureReviewWorkflow"]


# ---------------------------------------------------------------------------
# LLM 注入适配器
# ---------------------------------------------------------------------------
class _InjectedAuthenticator:
    """注入式 LLM 认证器。

    跳过 CSTCloud 网络认证, 直接返回构造时指定的 LLM 实例,
    使 :class:`LiteraturePipeline.run` 内部的 ``login()`` / ``get_llm()``
    调用全部走注入的模型 (如 Qwen3)。
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._authenticated = llm is not None

    def login(self, api_key: Optional[str] = None,
              base_url: Optional[str] = None) -> bool:
        return self._authenticated

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def get_llm(self):
        return self._llm


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _topic_slug(topic: str) -> str:
    """把研究主题转为文件名安全的小写 slug。"""
    slug = re.sub(r"[^\w\s-]", "", topic.strip().lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug or "research"


def _pipeline_report_to_dict(report: Any) -> dict:
    """把 :class:`PipelineReport` 转为报告生成器期望的 dict。

    报告生成器 (:class:`KnowledgeReportGenerator`) 的文献章节期望
    ``papers`` 为 dict 列表, 每项含 title/authors/year/relevance/summary。
    """
    papers = []
    for rec in getattr(report, "papers", []) or []:
        paper = getattr(rec, "paper", None)
        score = getattr(rec, "score", None)
        if paper is None:
            continue
        published = getattr(paper, "published_date", "") or ""
        papers.append({
            "title": getattr(paper, "title", "Untitled"),
            "authors": ", ".join(getattr(paper, "authors", [])[:5]) or "Unknown",
            "year": published[:4] if isinstance(published, str) else "",
            "relevance": getattr(score, "relevance", "—") if score else "—",
            "relevance_score": getattr(score, "score", 0.0) if score else 0.0,
            "summary": getattr(rec, "summary", None) or getattr(paper, "abstract", "") or "",
            "arxiv_id": getattr(paper, "id", ""),
            "status": getattr(rec, "status", ""),
        })
    return {
        "research_topic": getattr(report, "research_topic", ""),
        "total_found": getattr(report, "total_found", 0),
        "strong_count": getattr(report, "strong_count", 0),
        "weak_count": getattr(report, "weak_count", 0),
        "downloaded_count": getattr(report, "downloaded_count", 0),
        "papers": papers,
        "knowledge_graph_snapshot": getattr(report, "knowledge_graph_snapshot", {}) or {},
    }


# ---------------------------------------------------------------------------
# 工作流
# ---------------------------------------------------------------------------
@register_workflow()
class LiteratureReviewWorkflow(BaseWorkflow):
    """航天文献综述工作流。

    name = 'literature_review'

    一句提示词即可驱动::

        from aerospace_agent.workflows import execute_workflow
        r = execute_workflow("literature_review",
                             research_topic="lunar transfer orbit",
                             max_papers=5)
        print(r.summary)
        print(r.artifacts)  # [cloud_html, report_html]
    """

    name = "literature_review"
    description = (
        "航天文献综述自动化：搜索 arXiv 文献 → 相关性评分 → 下载/总结/索引 → "
        "知识图谱更新 → 知识云图 + 知识学习报告生成"
    )
    version = "1.0.0"
    required_tools: list = []  # 不依赖 mcp_tools, 依赖 rag/reporting 模块

    steps = [
        {"name": "init_pipeline", "description": "初始化 RAG、LLM 与文献处理管线"},
        {"name": "search_literature", "description": "搜索 arXiv 文献并按相关性评分"},
        {"name": "process_papers", "description": "处理强相关论文:下载/总结/索引/图谱更新"},
        {"name": "generate_cloud", "description": "生成交互式知识云图 HTML"},
        {"name": "generate_report", "description": "生成知识学习报告 HTML(内嵌云图)"},
        {"name": "finalize", "description": "汇总产出物与执行摘要"},
    ]

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> bool:
        if not super().validate_params(params):
            return False
        topic = params.get("research_topic", "lunar transfer orbit")
        if not isinstance(topic, str) or not topic.strip():
            return False
        max_papers = params.get("max_papers", 5)
        if not isinstance(max_papers, (int, float)) or max_papers <= 0:
            return False
        return True

    # ------------------------------------------------------------------
    # 主执行
    # ------------------------------------------------------------------
    def execute(
        self,
        research_topic: str = "lunar transfer orbit",
        max_papers: int = 5,
        min_relevance: str = "strong",
        llm: Optional[Any] = None,
        output_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        embed_cloud: bool = True,
        **kwargs,
    ) -> WorkflowResult:
        """执行文献综述工作流。

        Parameters
        ----------
        research_topic : str
            研究主题 (同时作为 arXiv 搜索关键词与相关性评分主题)。
        max_papers : int
            最大搜索/处理论文数。
        min_relevance : str
            最低相关性阈值, ``"strong"`` 或 ``"weak"``。
        llm : Any, optional
            注入的 LLM 实例 (如 Qwen3 ``LocalLLM``); 为 None 时回退到管线
            默认的认证/回退逻辑 (CSTCloud → create_llm → MockLLM)。
        output_dir : str, optional
            产出物 (云图/报告 HTML) 输出目录; 默认 ``<cwd>/reports``。
        data_dir : str, optional
            RAG 数据目录; None 时用默认持久化目录 (项目 ``data/``)。
        embed_cloud : bool
            报告是否内嵌知识云图 (iframe srcdoc, 自包含)。
        **kwargs
            额外参数 (如 cloud_title, report_title)。
        """
        res = WorkflowResult()
        res.metadata["params"] = {
            "research_topic": research_topic,
            "max_papers": max_papers,
            "min_relevance": min_relevance,
            "llm_injected": llm is not None,
            "llm_type": type(llm).__name__ if llm is not None else "default",
            "output_dir": output_dir,
            "data_dir": data_dir,
        }

        slug = _topic_slug(research_topic)
        out_dir = output_dir or os.path.join(os.getcwd(), "reports")
        cloud_path = os.path.join(out_dir, f"knowledge_cloud_{slug}.html")
        report_path = os.path.join(out_dir, f"literature_review_{slug}.html")
        cloud_title = kwargs.get("cloud_title", f"航天知识云图 — {research_topic}")

        # ---- 步骤 1: 初始化管线 ----
        pipeline = None
        try:
            pipeline = self._init_pipeline(llm=llm, data_dir=data_dir)
            llm_desc = (f"注入 LLM: {type(llm).__name__}"
                        if llm is not None else "默认 LLM (CSTCloud→create_llm→Mock)")
            self._log_step(
                res, "init_pipeline", "success",
                f"文献管线已初始化; {llm_desc}; 主题={research_topic!r}",
                data={"data_dir": data_dir, "llm_type": res.metadata["params"]["llm_type"]},
            )
        except Exception as exc:
            self._log_step(res, "init_pipeline", "failed", f"管线初始化失败: {exc}")
            res.summary = f"文献综述工作流初始化失败: {exc}"
            return res

        # ---- 步骤 2 & 3: 运行管线 (搜索 + 评分 + 处理) ----
        pipeline_report = None
        try:
            pipeline_report = pipeline.run(
                research_topic=research_topic,
                max_papers=max_papers,
                min_relevance=min_relevance,
            )
            total = getattr(pipeline_report, "total_found", 0)
            strong = getattr(pipeline_report, "strong_count", 0)
            downloaded = getattr(pipeline_report, "downloaded_count", 0)
            self._log_step(
                res, "search_literature", "success",
                f"搜索到 {total} 篇文献, 强相关 {strong} 篇",
                data={"total_found": total, "strong_count": strong,
                      "weak_count": getattr(pipeline_report, "weak_count", 0)},
            )
            self._log_step(
                res, "process_papers", "success",
                f"已下载并处理 {downloaded} 篇强相关论文 (总结/索引/图谱更新)",
                data={"downloaded_count": downloaded},
            )
        except Exception as exc:
            self._log_step(res, "search_literature", "failed",
                           f"管线运行失败: {exc}")
            self._log_step(res, "process_papers", "skipped",
                           "因搜索失败跳过论文处理")
            # 降级: 仍尝试基于现有知识图谱生成云图/报告
            pipeline_report = None

        # ---- 获取知识图谱 ----
        kg = self._get_knowledge_graph(pipeline)
        if kg is None:
            self._log_step(res, "generate_cloud", "skipped",
                           "知识图谱不可用, 跳过云图与报告生成")
            res.summary = (
                f"文献综述工作流完成 (降级): 主题={research_topic}, "
                f"但知识图谱不可用, 未生成云图/报告。"
            )
            res.success = pipeline_report is not None
            return res

        # ---- 步骤 4: 生成知识云图 ----
        cloud_gen = None
        try:
            from aerospace_agent.rag.knowledge_cloud import KnowledgeCloudGenerator
            cloud_gen = KnowledgeCloudGenerator()
            cloud_path = cloud_gen.generate(kg, output_path=cloud_path, title=cloud_title)
            self._log_step(
                res, "generate_cloud", "success",
                f"知识云图已生成: {cloud_path} (节点 {kg.num_nodes}, 边 {kg.num_edges})",
                data={"cloud_path": cloud_path,
                      "num_nodes": kg.num_nodes, "num_edges": kg.num_edges},
            )
            res.artifacts.append(cloud_path)
        except Exception as exc:
            self._log_step(res, "generate_cloud", "failed",
                           f"知识云图生成失败: {exc}")
            cloud_gen = None

        # ---- 步骤 5: 生成知识学习报告 ----
        try:
            from aerospace_agent.reporting.knowledge_report import KnowledgeReportGenerator
            report_gen = KnowledgeReportGenerator()
            pr_dict = _pipeline_report_to_dict(pipeline_report) if pipeline_report else None
            report_path = report_gen.generate(
                knowledge_graph=kg,
                pipeline_report=pr_dict,
                output_path=report_path,
                embed_cloud=embed_cloud,
                cloud_output_path=cloud_path if cloud_gen else None,
            )
            self._log_step(
                res, "generate_report", "success",
                f"知识学习报告已生成: {report_path}",
                data={"report_path": report_path,
                      "papers_count": len(pr_dict["papers"]) if pr_dict else 0,
                      "embed_cloud": embed_cloud},
            )
            res.artifacts.append(report_path)
        except Exception as exc:
            self._log_step(res, "generate_report", "failed",
                           f"知识学习报告生成失败: {exc}")

        # ---- 步骤 6: 汇总 ----
        res = self._finalize(res, research_topic, pipeline_report, kg)
        return res

    # ------------------------------------------------------------------
    # 内部: 初始化管线
    # ------------------------------------------------------------------
    @staticmethod
    def _init_pipeline(llm: Optional[Any] = None,
                       data_dir: Optional[str] = None):
        """创建 LiteraturePipeline, 并在提供 llm 时注入。"""
        from aerospace_agent.rag.aerospace_rag import AerospaceRAG
        from aerospace_agent.rag.literature_pipeline import LiteraturePipeline

        rag_kwargs = {}
        if data_dir is not None:
            rag_kwargs["data_dir"] = data_dir
        rag = AerospaceRAG(**rag_kwargs) if rag_kwargs else AerospaceRAG()

        pipeline = LiteraturePipeline(rag=rag)

        # 注入 LLM: 用适配器替换默认 authenticator, 使 run() 走注入模型
        if llm is not None:
            pipeline.authenticator = _InjectedAuthenticator(llm)
            pipeline.llm = llm
            # 同步给 scorer, run() 内部会再次设置, 此处确保提前一致
            pipeline.scorer.llm = llm
        return pipeline

    # ------------------------------------------------------------------
    # 内部: 获取知识图谱
    # ------------------------------------------------------------------
    @staticmethod
    def _get_knowledge_graph(pipeline):
        """从管线获取知识图谱实例, 失败返回 None。"""
        if pipeline is None:
            return None
        try:
            return pipeline.rag.kb.knowledge_graph
        except AttributeError:
            return None

    # ------------------------------------------------------------------
    # 内部: 汇总
    # ------------------------------------------------------------------
    def _finalize(self, res: WorkflowResult, topic: str,
                  pipeline_report, kg) -> WorkflowResult:
        """组装最终结果与摘要。"""
        total = getattr(pipeline_report, "total_found", 0) if pipeline_report else 0
        strong = getattr(pipeline_report, "strong_count", 0) if pipeline_report else 0
        downloaded = getattr(pipeline_report, "downloaded_count", 0) if pipeline_report else 0

        artifacts_str = ", ".join(res.artifacts) if res.artifacts else "无"
        res.success = bool(res.artifacts)  # 至少生成了一个产出物即视为成功
        res.result = {
            "research_topic": topic,
            "total_found": total,
            "strong_count": strong,
            "downloaded_count": downloaded,
            "knowledge_graph": {
                "num_nodes": kg.num_nodes if kg else 0,
                "num_edges": kg.num_edges if kg else 0,
            },
            "artifacts": list(res.artifacts),
        }
        res.metadata["total_found"] = total
        res.metadata["strong_count"] = strong
        res.metadata["downloaded_count"] = downloaded
        res.metadata["num_nodes"] = kg.num_nodes if kg else 0
        res.metadata["num_edges"] = kg.num_edges if kg else 0
        res.summary = (
            f"文献综述工作流完成: 主题={topic!r}, 搜索到 {total} 篇文献, "
            f"强相关 {strong} 篇, 已下载处理 {downloaded} 篇; "
            f"知识图谱 {kg.num_nodes if kg else 0} 节点/{kg.num_edges if kg else 0} 边; "
            f"产出物: {artifacts_str}."
        )

        self._log_step(
            res, "finalize", "success",
            res.summary,
            data={"success": res.success, "artifacts_count": len(res.artifacts)},
        )
        return res


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LiteratureReviewWorkflow 自测 (MockLLM, max_papers=3)")
    print("=" * 70)

    import tempfile

    wf = LiteratureReviewWorkflow()
    print(f"工作流: {wf!r}")
    print(f"步骤计划:")
    for line in wf.get_plan():
        print(f"  {line}")
    print(f"工具可用性: {wf.check_tools()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        out_dir = os.path.join(tmpdir, "reports")
        r = wf.execute(
            research_topic="lunar transfer orbit",
            max_papers=3,
            min_relevance="strong",
            output_dir=out_dir,
            data_dir=data_dir,
        )
        print(f"\nsuccess={r.success}")
        print(f"steps={len(r.steps_log)}")
        print(f"summary: {r.summary}")
        print(f"artifacts: {r.artifacts}")
        for s in r.steps_log:
            print(f"  [{s['status']}] {s['step']}: {s['detail']}")

    print("\nliterature_review 自测完成.")
