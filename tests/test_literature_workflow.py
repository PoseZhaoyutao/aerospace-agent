"""文献综述工作流 (literature_review) 测试。

测试分两部分:
  1. 离线测试 (默认运行): 工作流注册、参数校验、LLM 注入适配器、
     基于预填充知识图谱的云图/报告生成。
  2. Qwen3 集成测试 (@pytest.mark.qwen3): 注入 Qwen3 LocalLLM 运行完整
     文献综述工作流 (搜索 → 评分 → 下载/总结 → 云图 → 报告)。

运行方式:
  离线:  pytest tests/test_literature_workflow.py -v
  Qwen3: pytest -m qwen3 tests/test_literature_workflow.py -v
         (前置: d:\\Project\\Qwen3\\start_api.bat 启动 Qwen3 服务)

前置条件 (Qwen3):
  - Qwen3-VL-8B-Instruct API 运行在 http://127.0.0.1:8000
  - 可选: 网络可访问 arXiv (无网络时工作流降级为基于现有图谱生成云图/报告)
"""
from __future__ import annotations

import os

import pytest

QWEN3_URL = "http://127.0.0.1:8000/v1"
QWEN3_MODEL = "qwen3-vl"


# ===========================================================================
# 离线测试 (默认运行, 不依赖外部服务)
# ===========================================================================
class TestLiteratureWorkflowOffline:
    """工作流骨架与基础设施离线测试。"""

    def test_workflow_registered(self):
        """literature_review 应已注册到默认注册表。"""
        from aerospace_agent.workflows import list_workflow_names, get_workflow
        names = list_workflow_names()
        assert "literature_review" in names, f"literature_review 未注册: {names}"
        wf = get_workflow("literature_review")
        assert wf is not None
        assert wf.name == "literature_review"
        assert wf.version == "1.0.0"

    def test_workflow_steps(self):
        """工作流应含 6 个步骤。"""
        from aerospace_agent.workflows import get_workflow
        wf = get_workflow("literature_review")
        assert len(wf.steps) == 6
        step_names = [s["name"] for s in wf.steps]
        assert step_names == [
            "init_pipeline", "search_literature", "process_papers",
            "generate_cloud", "generate_report", "finalize",
        ]

    def test_workflow_info(self):
        """get_info 应返回完整元信息。"""
        from aerospace_agent.workflows import get_workflow
        wf = get_workflow("literature_review")
        info = wf.get_info()
        assert info["name"] == "literature_review"
        assert "文献" in info["description"]
        assert len(info["steps"]) == 6
        assert info["required_tools"] == []

    def test_get_plan(self):
        """步骤计划应按序号格式化。"""
        from aerospace_agent.workflows import get_workflow
        wf = get_workflow("literature_review")
        plan = wf.get_plan()
        assert len(plan) == 6
        assert plan[0].startswith("1. init_pipeline:")
        assert plan[-1].startswith("6. finalize:")

    def test_validate_params_valid(self):
        """合法参数应通过校验。"""
        from aerospace_agent.workflows import get_workflow
        wf = get_workflow("literature_review")
        assert wf.validate_params({"research_topic": "lunar transfer", "max_papers": 5})
        assert wf.validate_params({})  # 用默认值也应合法

    def test_validate_params_invalid(self):
        """非法参数应被拒绝。"""
        from aerospace_agent.workflows import get_workflow
        wf = get_workflow("literature_review")
        assert not wf.validate_params(None)
        assert not wf.validate_params("not a dict")
        assert not wf.validate_params({"research_topic": "", "max_papers": 5})
        assert not wf.validate_params({"research_topic": "x", "max_papers": 0})
        assert not wf.validate_params({"research_topic": "x", "max_papers": -1})

    def test_injected_authenticator_with_llm(self):
        """_InjectedAuthenticator 注入 LLM 后应认证成功。"""
        from aerospace_agent.workflows.literature_review import _InjectedAuthenticator

        class _FakeLLM:
            pass

        llm = _FakeLLM()
        auth = _InjectedAuthenticator(llm)
        assert auth.is_authenticated is True
        assert auth.login() is True
        assert auth.get_llm() is llm

    def test_injected_authenticator_without_llm(self):
        """_InjectedAuthenticator 未注入 LLM 应未认证。"""
        from aerospace_agent.workflows.literature_review import _InjectedAuthenticator
        auth = _InjectedAuthenticator(None)
        assert auth.is_authenticated is False
        assert auth.login() is False

    def test_topic_slug(self):
        """研究主题应转为文件名安全 slug。"""
        from aerospace_agent.workflows.literature_review import _topic_slug
        assert _topic_slug("Lunar Transfer Orbit") == "lunar_transfer_orbit"
        assert _topic_slug("a/b@c!") == "abc"
        assert _topic_slug("  multi   space  ") == "multi_space"
        assert _topic_slug("") == "research"

    def test_pipeline_report_to_dict(self):
        """_pipeline_report_to_dict 应正确转换 PipelineReport。"""
        from aerospace_agent.workflows.literature_review import _pipeline_report_to_dict

        class _FakeScore:
            relevance = "strong"
            score = 0.92

        class _FakePaper:
            id = "2401.00001"
            title = "Test Paper"
            authors = ["Author A", "Author B"]
            abstract = "Abstract text"
            published_date = "2024-01-15"

        class _FakeRec:
            paper = _FakePaper()
            score = _FakeScore()
            summary = "Summary text"
            status = "downloaded"

        class _FakeReport:
            research_topic = "test topic"
            total_found = 3
            strong_count = 2
            weak_count = 1
            downloaded_count = 1
            papers = [_FakeRec()]
            knowledge_graph_snapshot = {"num_nodes": 5}

        d = _pipeline_report_to_dict(_FakeReport())
        assert d["research_topic"] == "test topic"
        assert d["total_found"] == 3
        assert len(d["papers"]) == 1
        p = d["papers"][0]
        assert p["title"] == "Test Paper"
        assert p["authors"] == "Author A, Author B"
        assert p["year"] == "2024"
        assert p["relevance"] == "strong"
        assert p["summary"] == "Summary text"

    def test_cloud_and_report_offline(self, tmp_path):
        """离线验证: 基于预填充图谱生成云图与报告 (workflow 依赖的核心模块)。"""
        from aerospace_agent.rag.knowledge_graph import KnowledgeGraph
        from aerospace_agent.rag.knowledge_cloud import KnowledgeCloudGenerator
        from aerospace_agent.reporting.knowledge_report import KnowledgeReportGenerator

        kg = KnowledgeGraph()
        kg.prepopulate()
        assert kg.num_nodes > 0

        cloud_gen = KnowledgeCloudGenerator()
        cloud_path = cloud_gen.generate(
            kg, output_path=str(tmp_path / "cloud.html"),
            title="测试知识云图")
        assert os.path.isfile(cloud_path)
        with open(cloud_path, encoding="utf-8") as f:
            cloud_content = f.read()
        assert len(cloud_content) > 500, "云图 HTML 内容过短"

        report_gen = KnowledgeReportGenerator()
        report_path = report_gen.generate(
            kg, output_path=str(tmp_path / "report.html"))
        assert os.path.isfile(report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        assert "<html" in content.lower()
        assert "知识" in content


# ===========================================================================
# Qwen3 集成测试 (需 Qwen3 服务在线, 默认不运行)
# ===========================================================================
pytestmark_qwen3 = pytest.mark.qwen3


@pytest.mark.qwen3
@pytest.mark.slow
class TestLiteratureWorkflowQwen3:
    """注入 Qwen3 LLM 运行完整文献综述工作流。"""

    @pytest.fixture(scope="class")
    def qwen3_llm(self):
        """Qwen3 LocalLLM 实例 (模块级共享, 不可用则跳过)。"""
        from aerospace_agent.core.llm_interface import LocalLLM
        llm = LocalLLM(
            base_url=QWEN3_URL, model=QWEN3_MODEL,
            max_retries=3, retry_delay=2.0, timeout=600,
        )
        try:
            resp = llm.chat([{"role": "user", "content": "你好"}], max_tokens=10)
            assert resp, "Qwen3 返回空响应"
        except Exception as e:
            pytest.skip(f"Qwen3 服务不可用: {e}")
        return llm

    def test_qwen3_injected_workflow_run(self, qwen3_llm, tmp_path):
        """注入 Qwen3 运行文献综述工作流, 应成功并生成有效 HTML 产出物。

        使用 max_papers=1 + min_relevance='weak' 减少 LLM 调用
        (weak 论文仅记录元信息, 不触发下载/全文总结); 即使 arXiv
        网络不可用, 工作流降级路径也应生成云图+报告。
        """
        from aerospace_agent.workflows import execute_workflow
        r = execute_workflow(
            "literature_review",
            research_topic="lunar transfer orbit",
            max_papers=1,
            min_relevance="weak",
            llm=qwen3_llm,
            output_dir=str(tmp_path / "reports"),
            data_dir=str(tmp_path / "data"),
        )
        # 至少生成一个产出物即视为成功 (云图或报告)
        assert r.success, f"工作流应成功: {r.summary}"
        assert len(r.artifacts) >= 1, f"应至少生成一个产出物, artifacts={r.artifacts}"
        # LLM 注入标记应正确
        assert r.metadata["params"]["llm_injected"] is True
        assert r.metadata["params"]["llm_type"] == "LocalLLM"
        # 产出物文件应真实存在且为有效 HTML
        for art in r.artifacts:
            assert os.path.isfile(art), f"产出物文件不存在: {art}"
            with open(art, encoding="utf-8") as f:
                content = f.read()
            assert ("<html" in content.lower()
                    or "<!doctype" in content.lower()), f"{art} 应为 HTML"
            assert len(content) > 500, f"{art} 内容过短 ({len(content)} 字符)"
        # 步骤日志应记录关键步骤
        assert len(r.steps_log) >= 5, f"步骤日志过少: {len(r.steps_log)}"

    def test_qwen3_llm_directly(self, qwen3_llm):
        """Qwen3 应能生成结构化文献总结 (验证注入 LLM 的实际能力)。"""
        prompt = (
            "请用一句话总结: 霍曼转移轨道是航天器在两个共面圆轨道间"
            "转移的最节能方式。限 50 字以内。"
        )
        resp = qwen3_llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=120, temperature=0.3,
        )
        assert resp, "Qwen3 回复不应为空"
        assert any(kw in resp for kw in ["霍曼", "转移", "轨道", "Hohmann"]), \
            f"回复应包含相关关键词: {resp}"
