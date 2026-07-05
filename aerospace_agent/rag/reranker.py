"""规则重排序器 (Rule-based Reranker)。

向量+关键词+图谱融合后的分数已是不错的相关性估计, 但仍可叠加一些「领域
先验」规则把最该排在前面的结果提上来:

1. **公式优先**: 航天问答里, 公式类节点 (vis-viva / 开普勒方程) 通常是
   最有信息量的答案, 给 ``formula`` 类型加分。
2. **术语共享**: 结果文本与查询共享的关键术语越多, 越相关, 加分。
3. **多源命中**: 同时被向量+关键词(+图谱)检索到的结果更可靠, 加分
   (多路交叉印证)。
4. **图谱距离**: 图谱路命中的概念, 与查询命中概念距离越近越加分。

最终分数 = 融合分 * (1 + Σ规则奖励), 不改变相对量纲太多, 仅做温和重排。
"""

from __future__ import annotations

from typing import List

from .retriever import RetrievalResult

__all__ = ["Reranker"]


class Reranker:
    """基于规则的重排序器。"""

    # 各规则奖励上限 (相对原分的比例)
    BONUS_FORMULA = 0.15
    BONUS_TERM = 0.20
    BONUS_MULTISOURCE = 0.25
    BONUS_GRAPH_NEAR = 0.10

    def __init__(
        self,
        bonus_formula: float = BONUS_FORMULA,
        bonus_term: float = BONUS_TERM,
        bonus_multisource: float = BONUS_MULTISOURCE,
        bonus_graph_near: float = BONUS_GRAPH_NEAR,
    ):
        self.bonus_formula = bonus_formula
        self.bonus_term = bonus_term
        self.bonus_multisource = bonus_multisource
        self.bonus_graph_near = bonus_graph_near

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _terms(text: str) -> set:
        """极简取词: 小写后取连续字母数字/CJK 单字集合。"""
        terms = set()
        cur = ""
        for ch in text.lower():
            if "\u4e00" <= ch <= "\u9fff":
                if cur:
                    terms.add(cur)
                    cur = ""
                terms.add(ch)
            elif ch.isalnum() and ch.isascii():
                cur += ch
            else:
                if cur:
                    terms.add(cur)
                    cur = ""
        if cur:
            terms.add(cur)
        # 去极短英文词
        return {t for t in terms if len(t) > 1 or not t.isascii()}

    # ------------------------------------------------------------------ 重排
    def rerank(
        self, results: List[RetrievalResult], query: str
    ) -> List[RetrievalResult]:
        """对融合结果做规则重排, 返回新列表 (不就地修改)。"""
        q_terms = self._terms(query)
        reranked: List[RetrievalResult] = []
        for r in results:
            bonus = 0.0
            reasons: List[str] = []

            # 1) 公式优先
            ntype = r.metadata.get("node_type")
            if ntype == "formula" or r.metadata.get("type") == "formula":
                bonus += self.bonus_formula
                reasons.append("formula")

            # 2) 术语共享
            if q_terms:
                r_terms = self._terms(r.text)
                shared = q_terms & r_terms
                if shared:
                    ratio = len(shared) / max(len(q_terms), 1)
                    bonus += self.bonus_term * min(ratio, 1.0)
                    reasons.append(f"shared={len(shared)}")

            # 3) 多源命中
            sources = r.metadata.get("sources") or (
                [r.source] if r.source and "+" not in r.source else r.source.split("+")
            )
            n_sources = len({s for s in sources if s})
            if n_sources >= 2:
                bonus += self.bonus_multisource
                reasons.append(f"multi({n_sources})")

            # 4) 图谱近邻 (距离 0/1 的概念)
            if "graph" in sources and r.metadata.get("matched_concept"):
                # 距离信息若存在则用; 这里近邻奖励给所有图谱命中
                bonus += self.bonus_graph_near
                reasons.append("graph-near")

            new_score = r.score * (1.0 + bonus)
            # 把重排理由写入 explanation 便于观测
            extra = f"[rerank +{bonus:.2f} {'/'.join(reasons)}]" if reasons else ""
            rr = RetrievalResult(
                text=r.text,
                score=new_score,
                source=r.source,
                metadata=dict(r.metadata),
                explanation=(r.explanation + " " + extra).strip() if r.explanation else extra,
            )
            reranked.append(rr)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .retriever import RetrievalResult

    rs = [
        RetrievalResult(
            text="vis-viva 方程 v^2 = mu(2/r - 1/a) 是地月转移 delta-v 计算核心",
            score=0.30, source="vector+keyword",
            metadata={"sources": ["vector", "keyword"], "node_type": "formula"},
        ),
        RetrievalResult(
            text="orekit 是开源航天动力学库",
            score=0.32, source="vector",
            metadata={"sources": ["vector"]},
        ),
        RetrievalResult(
            text="Hohmann 转移用 vis-viva 算地月转移 delta-v",
            score=0.28, source="vector+keyword+graph",
            metadata={"sources": ["vector", "keyword", "graph"],
                      "matched_concept": "hohmann"},
        ),
    ]
    q = "地月转移 delta-v"
    print(f"[query] {q}")
    print("-- 融合原始排序 --")
    for r in sorted(rs, key=lambda x: x.score, reverse=True):
        print(f"  {r.score:.4f}  {r.text[:40]}")
    print("-- rerank 后 --")
    for r in Reranker().rerank(rs, q):
        print(f"  {r.score:.4f}  {r.text[:40]}  {r.explanation}")
