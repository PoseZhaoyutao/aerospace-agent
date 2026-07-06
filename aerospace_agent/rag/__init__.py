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
* :class:`KnowledgeCloudGenerator` 知识云图可视化生成器
* :class:`LiteraturePipeline`      文献处理管线 (搜索+评分+下载+索引+图谱)
* :class:`ArxivFetcher`            arXiv 文献搜索器
* :class:`CSTCloudAuthenticator`   中国科技云认证器
* :class:`RelevanceScorer`         摘要相关性评分器
* :class:`Paper`                   论文数据类
* :class:`PipelineReport`          管线运行报告
"""

from __future__ import annotations

from .aerospace_rag import AerospaceRAG
from .knowledge_base import AerospaceKnowledgeBase
from .knowledge_graph import KnowledgeGraph
from .knowledge_cloud import KnowledgeCloudGenerator
from .keyword_index import KeywordIndex
from .literature_fetcher import (
    ArxivFetcher,
    CSTCloudAuthenticator,
    Paper,
    extract_text_from_pdf,
)
from .literature_pipeline import (
    LiteraturePipeline,
    PaperRecord,
    PipelineReport,
)
from .relevance_scorer import RelevanceScorer, RelevanceScore
from .reranker import Reranker
from .retriever import HybridRetriever, RetrievalResult
from .vector_store import SimpleEmbedder, VectorStore

__all__ = [
    "AerospaceRAG",
    "KnowledgeGraph",
    "KnowledgeCloudGenerator",
    "VectorStore",
    "SimpleEmbedder",
    "KeywordIndex",
    "HybridRetriever",
    "RetrievalResult",
    "Reranker",
    "AerospaceKnowledgeBase",
    # 文献管线模块
    "LiteraturePipeline",
    "ArxivFetcher",
    "CSTCloudAuthenticator",
    "RelevanceScorer",
    "RelevanceScore",
    "Paper",
    "PaperRecord",
    "PipelineReport",
    "extract_text_from_pdf",
]

__version__ = "0.2.0"
