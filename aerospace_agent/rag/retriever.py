"""三路混合检索器 (Hybrid Retriever)。

把 :class:`VectorStore` (语义) + :class:`KeywordIndex` (词项 BM25) +
:class:`KnowledgeGraph` (概念依赖) 三路结果并行检索、各自归一化、加权融合。

为什么需要三路
--------------
* **向量路** 擅长「意思相近」(语义模糊匹配), 但对生僻术语、公式符号弱;
* **关键词路** 擅长「精确命中」稀有术语 (BM25 的 IDF 让稀有词权重大),
  但不懂同义/近义;
* **图谱路** 擅长「概念联想」: 即便查询只提了「地月转移」, 也能沿
  ``depends_on`` 主动召回 Hohmann / vis-viva / 拼凑圆锥 / 发射窗口 等推导链
  上的概念, 这是单纯字面/语义检索都做不到的。

融合策略
--------
1. 三路各自取 ``top_k`` (实际取更大候选集 ``fetch_k = top_k * 3`` 以提升召回);
2. 每路分数按各自最大值归一化到 [0, 1];
3. 同一文档/概念按「文本」去重, 累加各路加权分数;
4. 记录命中来源 (供 :class:`Reranker` 多源加分)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .keyword_index import KeywordIndex
from .knowledge_graph import KnowledgeGraph
from .vector_store import VectorStore

__all__ = ["RetrievalResult", "HybridRetriever"]


@dataclass
class RetrievalResult:
    """单条检索结果。"""

    text: str
    score: float
    source: str  # 'vector' / 'keyword' / 'graph' / 多源时 'vector+keyword' 等
    metadata: dict = field(default_factory=dict)
    explanation: str = ""

    def __repr__(self) -> str:  # 便于调试
        return (
            f"RetrievalResult(score={self.score:.4f}, source={self.source}, "
            f"text={self.text[:40]!r}...)"
        )


class HybridRetriever:
    """三路混合检索器。"""

    # 图谱 BFS 邻域取概念解释的深度
    GRAPH_DEPTH = 2
    # 距离衰减: distance -> 权重
    _DIST_DECAY = {0: 1.0, 1: 0.6, 2: 0.35, 3: 0.2}

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        knowledge_graph: KnowledgeGraph,
    ):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.knowledge_graph = knowledge_graph

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _max_normalize(pairs: List[Tuple[float, str, dict, str]]) -> List[Tuple[float, str, dict, str]]:
        """把 [(raw_score, text, metadata, source)] 按 max 归一化到 [0,1]。"""
        if not pairs:
            return []
        mx = max(p[0] for p in pairs)
        if mx <= 0:
            return [(0.0, t, m, s) for (r, t, m, s) in pairs]
        return [(r / mx, t, m, s) for (r, t, m, s) in pairs]

    def _doc_text_meta(self, doc_id: int) -> Tuple[str, dict]:
        """由 doc_id 取 (text, metadata), 与向量库对齐。"""
        if 0 <= doc_id < len(self.vector_store.texts):
            return self.vector_store.texts[doc_id], self.vector_store.metadatas[doc_id]
        return "", {}

    # ------------------------------------------------------------------ 三路
    def _vector_candidates(self, query: str, fetch_k: int):
        out = []
        for score, text, meta in self.vector_store.search(query, top_k=fetch_k):
            out.append((score, text, dict(meta), "vector"))
        return out

    def _keyword_candidates(self, query: str, fetch_k: int):
        out = []
        for score, doc_id in self.keyword_index.search(query, top_k=fetch_k):
            text, meta = self._doc_text_meta(doc_id)
            if not text:
                continue
            m = dict(meta)
            m["doc_id"] = doc_id
            out.append((score, text, m, "keyword"))
        return out

    def _graph_candidates(self, query: str, fetch_k: int):
        """图谱路: 识别 query 命中概念 -> BFS 邻域 -> 概念解释作为候选。"""
        out = []
        hits = self.knowledge_graph.match_concepts(query)
        if not hits:
            return out
        # 每个命中概念展开邻域
        seen_node: Dict[str, float] = {}  # node_id -> best raw score
        matched_by: Dict[str, str] = {}   # node_id -> 命中它的概念
        for concept_id, match_score in hits[:5]:
            sub = self.knowledge_graph.query(concept_id, depth=self.GRAPH_DEPTH)
            for nid in sub["nodes"]:
                d = sub["distances"].get(nid, 99)
                decay = self._DIST_DECAY.get(d, 0.1)
                raw = match_score * decay
                if nid not in seen_node or raw > seen_node[nid]:
                    seen_node[nid] = raw
                    matched_by[nid] = concept_id
        # 构造候选文本 = 节点 content
        for nid, raw in seen_node.items():
            node = self.knowledge_graph.nodes[nid]
            text = node["content"]
            meta = {
                "node_id": nid,
                "node_type": node["type"],
                "matched_concept": matched_by[nid],
                "kg": True,
            }
            out.append((raw, text, meta, "graph"))
        # 按原始分排序取前 fetch_k
        out.sort(key=lambda x: x[0], reverse=True)
        return out[:fetch_k]

    # ------------------------------------------------------------------ 主入口
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        use_reranker: bool = False,
    ) -> List[RetrievalResult]:
        """三路并行检索 + 归一化 + 加权融合。

        参数
        ----
        query       : 查询文本
        top_k       : 最终返回数量
        weights     : (vector, keyword, graph) 三路权重, 无需归一化
        use_reranker: 是否额外套用规则重排 (见 :mod:`reranker`)
        """
        w_vec, w_kw, w_graph = weights
        fetch_k = max(top_k * 3, 12)

        vec_c = self._max_normalize(self._vector_candidates(query, fetch_k))
        kw_c = self._max_normalize(self._keyword_candidates(query, fetch_k))
        graph_c = self._max_normalize(self._graph_candidates(query, fetch_k))

        # 按「归一化文本」去重合并: key -> {score, sources, meta, text, explanation}
        merged: Dict[str, dict] = {}

        def _add(cands, weight):
            for norm_score, text, meta, src in cands:
                key = self._norm_key(text)
                if not key:
                    continue
                contribution = weight * norm_score
                if key not in merged:
                    merged[key] = {
                        "text": text,
                        "score": 0.0,
                        "sources": [],
                        "meta": dict(meta),
                        "explanations": [],
                    }
                e = merged[key]
                e["score"] += contribution
                if src not in e["sources"]:
                    e["sources"].append(src)
                # 图谱候选带推导链解释
                if src == "graph" and meta.get("matched_concept"):
                    chain = self.knowledge_graph.explain(meta["matched_concept"])
                    e["explanations"].append(
                        f"[图谱: {meta['matched_concept']}→{meta['node_id']}] "
                        f"{chain.splitlines()[0]}"
                    )
                # 合并 metadata (图谱节点信息优先保留)
                if src == "graph":
                    e["meta"].update(meta)

        _add(vec_c, w_vec)
        _add(kw_c, w_kw)
        _add(graph_c, w_graph)

        results: List[RetrievalResult] = []
        for e in merged.values():
            srcs = e["sources"]
            source = "+".join(srcs) if len(srcs) > 1 else (srcs[0] if srcs else "")
            explanation = " | ".join(e["explanations"]) if e["explanations"] else ""
            meta = e["meta"]
            meta["sources"] = srcs
            results.append(
                RetrievalResult(
                    text=e["text"],
                    score=e["score"],
                    source=source,
                    metadata=meta,
                    explanation=explanation,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)

        if use_reranker:
            from .reranker import Reranker  # 延迟导入避免循环
            results = Reranker().rerank(results, query)

        return results[:top_k]

    @staticmethod
    def _norm_key(text: str) -> str:
        """文本归一化键: 去空白/标点/大小写, 用于跨源去重。"""
        import re as _re

        return _re.sub(r"[\s\W_]+", "", text).lower()

    # ------------------------------------------------------------------ 单路调试
    def retrieve_vector(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        return [
            RetrievalResult(text=t, score=s, source="vector", metadata=m)
            for s, t, m in self.vector_store.search(query, top_k)
        ]

    def retrieve_keyword(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        res = []
        for s, did in self.keyword_index.search(query, top_k):
            t, m = self._doc_text_meta(did)
            mm = dict(m); mm["doc_id"] = did
            res.append(RetrievalResult(text=t, score=s, source="keyword", metadata=mm))
        return res

    def retrieve_graph(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        cands = self._graph_candidates(query, top_k)
        cands = self._max_normalize(cands)
        res = []
        for norm_score, text, meta, _src in cands:
            exp = ""
            if meta.get("matched_concept"):
                exp = self.knowledge_graph.explain(meta["matched_concept"]).splitlines()[0]
            res.append(
                RetrievalResult(
                    text=text, score=norm_score, source="graph",
                    metadata=meta, explanation=exp,
                )
            )
        return res[:top_k]


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .keyword_index import KeywordIndex
    from .knowledge_graph import KnowledgeGraph
    from .vector_store import VectorStore

    vs = VectorStore()
    ki = KeywordIndex()
    kg = KnowledgeGraph()
    kg.prepopulate()

    docs = [
        "Hohmann 转移是共面圆轨道间最省能量的双脉冲转移, 用 vis-viva 算 delta-v",
        "vis-viva 方程 v^2 = mu (2/r - 1/a) 由能量守恒导出, 是 delta-v 计算核心",
        "地月转移通常用拼凑圆锥近似, 分地心段与月心段, 需计算发射窗口相位角",
        "Lambert 问题已知两点位置与飞行时间求解轨道, 是 Hohmann 的通用化",
        "开普勒方程 M = E - e sinE 描述平近点角与偏近点角关系",
        "orekit 是 CNES 开源航天动力学库, 支持二体与数值传播",
    ]
    for i, t in enumerate(docs):
        vs.add(t, {"doc_id": i, "topic": t[:12]})
        ki.add(i, t)
    vs.reindex()

    retriever = HybridRetriever(vs, ki, kg)

    q = "地月转移用什么方法"
    print(f"=== 混合检索: {q} ===")
    for r in retriever.retrieve(q, top_k=5):
        print(f"  [{r.source:>12}] {r.score:.4f}  {r.text[:50]}")
        if r.explanation:
            print(f"               ↳ {r.explanation[:80]}")

    print(f"\n=== 三路对比 (各 top3): {q} ===")
    for name, fn in [("vector", retriever.retrieve_vector),
                     ("keyword", retriever.retrieve_keyword),
                     ("graph", retriever.retrieve_graph)]:
        print(f"-- {name} --")
        for r in fn(q, top_k=3):
            print(f"   {r.score:.4f}  {r.text[:50]}")
