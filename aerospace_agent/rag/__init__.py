"""aerospace_agent.rag — 航天知识检索增强生成 (RAG) 子包。

三路混合检索 (向量库 + 关键词倒排 + 知识图谱), 纯 numpy/scipy 实现,
不依赖外部向量库 / 预训练模型。

主要导出
--------
* :class:`AerospaceRAG`     顶层门面 (对外统一接口)
* :class:`KnowledgeGraph`   航天知识图谱 (核心创新)
* :class:`VectorStore`      向量存储 (n-gram + TF-IDF + 随机投影)
* :class:`KeywordIndex`     关键词倒排索引 (BM25)
* :class:`HybridRetriever`  三路混合检索器
* :class:`Reranker`         规则重排器
* :class:`AerospaceKnowledgeBase` 知识库管理
* :class:`RetrievalResult`  检索结果数据类
* :class:`SimpleEmbedder`   嵌入器
"""

from __future__ import annotations

from .aerospace_rag import AerospaceRAG
from .knowledge_base import AerospaceKnowledgeBase
from .knowledge_graph import KnowledgeGraph
from .keyword_index import KeywordIndex
from .reranker import Reranker
from .retriever import HybridRetriever, RetrievalResult
from .vector_store import SimpleEmbedder, VectorStore

__all__ = [
    "AerospaceRAG",
    "KnowledgeGraph",
    "VectorStore",
    "SimpleEmbedder",
    "KeywordIndex",
    "HybridRetriever",
    "RetrievalResult",
    "Reranker",
    "AerospaceKnowledgeBase",
]

__version__ = "0.1.0"
