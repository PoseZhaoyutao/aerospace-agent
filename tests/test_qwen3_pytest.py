"""Qwen3 本地模型集成测试（pytest 格式）。

标记为 @pytest.mark.qwen3，默认不运行（需要 Qwen3 服务在线）。
运行方式: pytest -m qwen3 tests/test_qwen3_pytest.py -v

前置条件:
  - Qwen3-VL-8B-Instruct API 服务运行在 http://127.0.0.1:8000
  - 启动方式: d:\\Project\\Qwen3\\start_api.bat
"""
from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.qwen3

QWEN3_URL = "http://127.0.0.1:8000/v1"
QWEN3_MODEL = "qwen3-vl"


@pytest.fixture(scope="module")
def qwen3_llm():
    """Qwen3 LocalLLM 实例（模块级共享）。"""
    from aerospace_agent.core.llm_interface import LocalLLM
    llm = LocalLLM(
        base_url=QWEN3_URL,
        model=QWEN3_MODEL,
        max_retries=3,
        retry_delay=2.0,
    )
    # 验证连接
    try:
        resp = llm.chat([
            {"role": "user", "content": "你好"},
        ], max_tokens=10)
        assert resp, "Qwen3 返回空响应"
    except Exception as e:
        pytest.skip(f"Qwen3 服务不可用: {e}")
    return llm


@pytest.fixture
def qwen3_agent(qwen3_llm):
    """使用 Qwen3 LLM 的完整 Agent。"""
    from aerospace_agent.core.agent import create_default_agent
    agent = create_default_agent(force_mock=False, max_steps=6)
    agent.llm = qwen3_llm
    return agent


class TestQwen3Basic:
    """Qwen3 基础对话测试。"""

    def test_qwen3_responds(self, qwen3_llm):
        """Qwen3 应能正常回复。"""
        resp = qwen3_llm.chat([
            {"role": "system", "content": "你是航天助手，简短回答。"},
            {"role": "user", "content": "开普勒第三定律是什么？一句话。"},
        ], max_tokens=100, temperature=0.3)
        assert resp, "回复不应为空"
        assert any(kw in resp for kw in ["开普勒", "定律", "周期", "半长轴"]), \
            f"回复应包含关键词: {resp}"


class TestQwen3ReAct:
    """Qwen3 ReAct 循环测试。"""

    def test_orbital_velocity_tool(self, qwen3_agent):
        """Agent 应通过 ReAct 调用 orbital_velocity 工具。"""
        task = "请使用 orbital_velocity 工具计算 400km 高度圆轨道的轨道速度。先调用工具，再根据结果给出 Final Answer。"
        result = qwen3_agent.run(task)
        assert "7.6" in str(result) or "7.7" in str(result), \
            f"预期 ~7.67 km/s，实际: {result}"

    def test_calculator_tool(self, qwen3_agent):
        """Agent 应通过 ReAct 调用 calculator 工具。"""
        task = "请使用 calculator 工具计算表达式 2*math.pi*math.sqrt(6778137**3/398600441800000) ，然后给出 Final Answer。"
        result = qwen3_agent.run(task)
        assert "5553" in str(result) or "5554" in str(result), \
            f"预期 ~5553s，实际: {result}"

    def test_tli_delta_v(self, qwen3_agent):
        """Agent 应计算地月转移 TLI 速度增量。"""
        task = "计算从 400km 停泊轨道到月球转移轨道(TLI)的速度增量。请先用 orbit_calculator 工具计算，然后给出 Final Answer。"
        result = qwen3_agent.run(task)
        assert "3.0" in str(result) or "3.1" in str(result), \
            f"预期 ~3.1 km/s，实际: {result}"

    def test_react_multi_step(self, qwen3_agent):
        """Agent 应支持多步 ReAct（先查速度再算周期）。"""
        task = ("先用 orbital_velocity 工具计算 500km 圆轨道速度，"
                "再用 calculator 工具计算 2*math.pi*6878137/结果数值 得到周期（秒），"
                "最后给出 Final Answer。")
        result = qwen3_agent.run(task)
        # 500km 圆轨道速度约 7.61 km/s, 周期约 5677s
        assert result, "应返回非空结果"


class TestQwen3RAG:
    """Qwen3 + RAG 检索测试。"""

    def test_rag_query_hohmann(self, qwen3_agent):
        """RAG 应检索到霍曼转移相关知识。"""
        assert qwen3_agent.rag is not None, "RAG 应可用"
        result = qwen3_agent.rag.query("霍曼转移轨道", top_k=3)
        assert any(kw in result for kw in ["霍曼", "Hohmann", "转移"]), \
            f"应包含霍曼转移关键词: {result[:200]}"

    def test_rag_routed_multisource(self, qwen3_agent):
        """RAG 多源路由应返回结果。"""
        rag = qwen3_agent.rag
        if not hasattr(rag, 'router'):
            pytest.skip("增强 RAG 不可用")
        result = rag.query_routed("轨道力学 开普勒", top_k=3)
        assert "results" in result
        assert len(result["results"]) > 0, "应返回检索结果"
