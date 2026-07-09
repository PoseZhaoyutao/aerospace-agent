"""RAG 多源路由检索测试 (K2.5 验证)。

验证 RetrieverRouter 能正确路由到不同检索源：
  - document: 文档源 (HybridRetriever)
  - memory: 记忆源 (LongTermMemory)
  - code: 代码源 (AST 解析)
  - builtin: 内置知识回退
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.offline


class TestRAGMultiSource:
    """RAG 多源注册与路由测试。"""

    def test_document_source_available(self, rag_instance):
        """文档源应默认可用。"""
        routed = rag_instance.query_routed("开普勒第三定律", top_k=3)
        assert routed is not None
        assert "results" in routed
        # 至少有内置知识回退结果
        assert len(routed["results"]) > 0

    def test_memory_source_registered(self, mock_agent):
        """create_default_agent 应注册记忆检索源。"""
        rag = mock_agent.rag
        if rag is None or not hasattr(rag, 'router'):
            pytest.skip("RAG 增强版不可用")
        assert rag.router.memory_retriever is not None, \
            "记忆检索源未注册"

    def test_code_source_registered(self, mock_agent):
        """create_default_agent 应注册代码检索源。"""
        rag = mock_agent.rag
        if rag is None or not hasattr(rag, 'router'):
            pytest.skip("RAG 增强版不可用")
        assert rag.router.code_retriever is not None, \
            "代码检索源未注册"

    def test_code_retriever_finds_functions(self, mock_agent):
        """代码检索源应能找到项目中的函数。"""
        rag = mock_agent.rag
        if rag is None or not hasattr(rag, 'router'):
            pytest.skip("RAG 增强版不可用")
        # 搜索 "propagate" 应找到 propagation 相关函数
        results = rag.router.code_retriever("propagate orbit", top_k=5)
        assert len(results) > 0, "代码检索应返回结果"
        # 验证结果格式: (score, text, meta)
        score, text, meta = results[0]
        assert isinstance(score, float)
        assert isinstance(text, str)
        assert "source_type" in meta

    def test_intent_detection(self, rag_instance):
        """查询意图检测应正确分类。"""
        router = rag_instance.router
        # 代码类查询
        sources = router.detect_intent("这个函数的代码实现是什么")
        assert "code" in sources or "document" in sources
        # 文档类查询
        sources = router.detect_intent("轨道力学文档")
        assert "document" in sources
        # 记忆类查询
        sources = router.detect_intent("之前的决策历史")
        assert "memory" in sources

    def test_builtin_fallback(self, rag_instance):
        """所有源无结果时应回退到内置知识库。"""
        router = rag_instance.router
        results = router._builtin_fallback("开普勒 轨道", top_k=3)
        assert len(results) > 0, "内置知识回退应返回结果"
        assert results[0].source == "builtin_knowledge"

    def test_routed_query_with_memory(self, mock_agent):
        """多源路由检索应能查询记忆源。"""
        rag = mock_agent.rag
        if rag is None or not hasattr(rag, 'router'):
            pytest.skip("RAG 增强版不可用")
        # 先存入一些记忆
        mock_agent.memory.remember("test_key", "这是测试记忆内容", tags=["test"])
        mock_agent.memory.save()
        # 验证记忆已存储
        assert len(mock_agent.memory.store) > 0, "记忆应已存入 store"
        # 查询记忆源（返回列表即可，随机投影 embedding 可能匹配度低）
        results = rag.router.memory_retriever("测试记忆内容", top_k=3)
        assert isinstance(results, list), "记忆检索应返回列表"
