"""关键词倒排索引 (Keyword Inverted Index) + BM25 排序。

本模块实现第二路检索: 基于词项的精确匹配 + BM25 相关性打分。

    * :class:`KeywordIndex` — 倒排表 ``{term: [(doc_id, tf), ...]}`` + BM25 打分。

分词策略与 :mod:`vector_store` 一致: 中文按字、英文按词; 英文词做极简
stemming (去常见后缀 -ing/-ed/-es/-s), 以提升召回。

BM25 公式推导 (见 :meth:`KeywordIndex.search` 的注释)。
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

__all__ = ["KeywordIndex"]

# 极简英文停用词 (避免高频无义字主导打分)
_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
    "are", "be", "by", "with", "as", "at", "it", "this", "that", "from",
    "the", "的", "了", "是", "在", "用", "与", "和", "及", "或", "为", "对",
}


def _simple_stem(word: str) -> str:
    """极简英文词干提取: 去掉常见后缀, 让 transfers/transfered 都归一到 transfer。

    这不是真正的 Porter stemmer, 但对航天术语足够 (且零依赖)。
    """
    if len(word) <= 3:
        return word
    for suf in ("ization", "ization", "ations", "ation", "ations"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    for suf in ("ing", "ies", "ied", "es", "ed", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf in ("ies", "ied"):
                return word[: -len(suf)] + "y"
            return word[: -len(suf)]
    return word


# 连续 ASCII 字母数字
_ASCII_WORD = re.compile(r"[a-z0-9]+")


class KeywordIndex:
    """倒排索引 + BM25 检索。

    数据结构
    --------
    * ``inverted``   : ``{term: [(doc_id, tf), ...]}`` 倒排表
    * ``doc_len``    : ``{doc_id: length}`` 每篇文档长度 (词项数)
    * ``doc_terms``  : ``{doc_id: {term: tf}}`` 每篇文档的词项频率 (便于打分)
    * ``N``          : 文档总数
    * ``avgdl``      : 平均文档长度
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # BM25 调参常数: k1 控制词频饱和, b 控制文档长度归一化强度
        self.k1 = float(k1)
        self.b = float(b)
        self.inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len: Dict[int, int] = {}
        self.doc_terms: Dict[int, Dict[str, int]] = {}
        self.doc_ids: Set[int] = set()

    # ------------------------------------------------------------------ 分词
    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词: 中文按字、英文按词 + 极简 stemming + 去停用词。"""
        if not text:
            return []
        text = text.lower()
        tokens: List[str] = []
        cur = ""
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                if cur:
                    stemmed = _simple_stem(cur)
                    if stemmed not in _STOPWORDS:
                        tokens.append(stemmed)
                    cur = ""
                if ch not in _STOPWORDS:
                    tokens.append(ch)
            elif ch.isascii() and ch.isalnum():
                cur += ch
            else:
                if cur:
                    stemmed = _simple_stem(cur)
                    if stemmed not in _STOPWORDS:
                        tokens.append(stemmed)
                    cur = ""
        if cur:
            stemmed = _simple_stem(cur)
            if stemmed not in _STOPWORDS:
                tokens.append(stemmed)
        return tokens

    # ------------------------------------------------------------------ 建索引
    def add(self, doc_id: int, text: str) -> None:
        """把 (doc_id, text) 加入倒排索引。重复 doc_id 会覆盖旧内容。"""
        # 若已存在, 先移除旧记录
        if doc_id in self.doc_ids:
            self._remove(doc_id)
        tokens = self._tokenize(text)
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        self.doc_len[doc_id] = len(tokens)
        self.doc_terms[doc_id] = dict(tf)
        self.doc_ids.add(doc_id)
        for term, freq in tf.items():
            self.inverted[term].append((doc_id, freq))

    def _remove(self, doc_id: int) -> None:
        """从倒排表中移除某文档。"""
        for term in list(self.doc_terms.get(doc_id, {}).keys()):
            self.inverted[term] = [
                (d, f) for (d, f) in self.inverted.get(term, []) if d != doc_id
            ]
            if not self.inverted[term]:
                del self.inverted[term]
        self.doc_len.pop(doc_id, None)
        self.doc_terms.pop(doc_id, None)
        self.doc_ids.discard(doc_id)

    def _avgdl(self) -> float:
        if not self.doc_len:
            return 0.0
        return sum(self.doc_len.values()) / len(self.doc_len)

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    # ------------------------------------------------------------------ BM25
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, int]]:
        """BM25 检索。返回 [(score, doc_id), ...] 按分数降序。

        BM25 打分公式推导
        -----------------
        对查询 Q = {q1, q2, ...} 与文档 D, 总分:

            score(D, Q) = Σ_{q ∈ Q}  IDF(q) · TF_term(q, D)

        其中:

        1) 词频饱和项 (Saturation):
                   f(q, D) · (k1 + 1)
            TF = ─────────────────────────────
                 f(q, D) + k1 · (1 - b + b · |D| / avgdl)

           f(q,D) 为 q 在 D 中的频次; |D| 为 D 长度; avgdl 为平均文档长度。
           k1 控制词频饱和速度 (越大越接近线性); b 控制长度归一化强度 (0=不归一,
           1=完全归一)。本项目取 k1=1.5, b=0.75 (经验默认值)。

        2) 逆文档频率 (采用带平滑的 Robertson-Sparck Jones 形式, 保证非负):
                   N - n(q) + 0.5
            IDF = ln( ─────────────── + 1 )
                     n(q) + 0.5
           N 为文档总数, n(q) 为含 q 的文档数。+1 在 log 内保证 IDF ≥ 0
           (经典 BM25 的 IDF 在 n(q)>N/2 时会变负, 加 1 修正)。

        直觉: 出现越稀有的词命中越重要 (IDF 大); 同一文档里词频越高越相关,
        但有上限 (饱和); 长文档需惩罚 (b·|D|/avgdl)。
        """
        if not self.doc_ids:
            return []
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        N = self.num_docs
        avgdl = self._avgdl() or 1.0
        scores: Dict[int, float] = defaultdict(float)

        # 查询词去重计数, 让重复查询词仍只算一次 (标准 BM25 做法)
        uniq_terms = set(query_terms)
        for q in uniq_terms:
            postings = self.inverted.get(q)
            if not postings:
                continue
            n_q = len(postings)  # 含 q 的文档数
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            for doc_id, f in postings:
                dl = self.doc_len[doc_id]
                tf_norm = (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                )
                scores[doc_id] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(float(s), d) for d, s in ranked[:top_k]]

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str) -> None:
        """以 JSON 持久化到 path。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "inverted": dict(self.inverted),
            "doc_len": {str(k): v for k, v in self.doc_len.items()},
            "doc_terms": {
                str(k): v for k, v in self.doc_terms.items()
            },
            "doc_ids": sorted(self.doc_ids),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从 JSON 加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = float(data.get("k1", 1.5))
        self.b = float(data.get("b", 0.75))
        self.inverted = defaultdict(
            list, {k: [tuple(x) for x in v] for k, v in data.get("inverted", {}).items()}
        )
        self.doc_len = {int(k): v for k, v in data.get("doc_len", {}).items()}
        self.doc_terms = {
            int(k): v for k, v in data.get("doc_terms", {}).items()
        }
        self.doc_ids = set(data.get("doc_ids", []))

    def clear(self) -> None:
        self.inverted = defaultdict(list)
        self.doc_len = {}
        self.doc_terms = {}
        self.doc_ids = set()


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    idx = KeywordIndex()
    docs = {
        0: "Hohmann 转移是共面圆轨道间最省能量的双脉冲转移, 用 vis-viva 算 delta-v",
        1: "vis-viva 方程 v^2 = mu (2/r - 1/a) 由能量守恒导出",
        2: "地月转移通常用拼凑圆锥近似, 需要计算发射窗口的相位角",
        3: "Lambert 问题已知两点位置与飞行时间求解轨道",
        4: "开普勒方程 M = E - e sinE, 平近点角与偏近点角的关系",
    }
    for did, txt in docs.items():
        idx.add(did, txt)

    q = "地月转移发射窗口"
    print(f"[query] {q}")
    for score, did in idx.search(q, top_k=3):
        print(f"  {score:.4f}  doc{did}  {docs[did]}")

    # 持久化
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ki.json")
        idx.save(p)
        idx2 = KeywordIndex()
        idx2.load(p)
        r1 = idx.search(q, top_k=3)
        r2 = idx2.search(q, top_k=3)
        print(f"[ok] 持久化: 保存/加载后结果一致 = {r1 == r2}")
