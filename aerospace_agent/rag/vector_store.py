"""向量存储 (Vector Store)。

本模块实现一个 **纯 numpy / scipy** 的向量检索后端, 不依赖任何外部向量库:

    * :class:`SimpleEmbedder` — 用「字符级 n-gram (n=3) + TF-IDF 权重 + 随机投影」
      把文本映射到 256 维稠密向量。无需预训练模型, 仅靠固定随机种子 (seed=42)
      保证嵌入稳定可复现。
    * :class:`VectorStore` — 基于余弦相似度的向量检索库, 支持 add / search /
      save / load / clear。

设计要点
--------
1. **可复现性**: 随机投影矩阵由 ``np.random.RandomState(42)`` 生成, 词项到桶的
   映射用确定性 FNV-1a 哈希 (不依赖 Python 进程级 hash 随机化)。因此「相同文本
   在相同索引状态下 → 相同向量」成立。
2. **中英文混合分词**: 中文按「字」切分, 英文按「词」(连续 ASCII 字母数字) 切分,
   保留原始顺序, 以便构造有意义的 n-gram。
3. **特征**: 对 token 序列抽取 1/2/3-gram (n=3 为主), 哈希到固定词表桶
   (vocab_size = 2^15), 再做 sublinear TF * IDF 加权。
4. **随机投影**: 高维稀疏特征向量 x ∈ R^V 经高斯随机矩阵 R ∈ R^{V×256} 投影为
   稠密向量 ``emb = xᵀ R``, 再做 L2 归一化。投影保留近邻关系 (Johnson-
   Lindenstrauss 思想), 且维度固定 256, 便于余弦检索。
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["SimpleEmbedder", "VectorStore"]


# ---------------------------------------------------------------------------
# 确定性哈希 (FNV-1a 32 bit)。Python 内建 hash() 对字符串做了进程级随机化,
# 不能用于需要跨进程复现的嵌入, 因此这里实现一个稳定的哈希。
# ---------------------------------------------------------------------------
def _stable_hash(s: str) -> int:
    """对字符串做确定性 32 位 FNV-1a 哈希。"""
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# 匹配「连续 ASCII 字母数字」作为一个英文/数字词
_ASCII_WORD = re.compile(r"[a-z0-9]+")


class SimpleEmbedder:
    """字符级 n-gram + TF-IDF + 随机投影 嵌入器。

    参数
    ----
    dim        : 输出向量维度 (默认 256)。
    ngram      : n-gram 最大阶数 (默认 3, 即 1/2/3-gram)。
    vocab_size : 哈希词表桶数 (默认 2^15 = 32768)。
    seed       : 随机投影矩阵种子 (默认 42, 保证可复现)。
    """

    def __init__(
        self,
        dim: int = 256,
        ngram: int = 3,
        vocab_size: int = 1 << 15,
        seed: int = 42,
    ):
        self.dim = int(dim)
        self.ngram = int(ngram)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)

        # 高斯随机投影矩阵, 固定 seed -> 跨进程可复现。
        # 除以 sqrt(dim) 让投影后方差与输入量级匹配 (归一化前数值稳定)。
        rng = np.random.RandomState(self.seed)
        self.proj = (rng.randn(self.vocab_size, self.dim) / math.sqrt(self.dim)).astype(
            np.float32
        )

        # 语料级统计量, 用于 IDF。add 时更新, search 时不更新。
        self.df: Dict[str, int] = {}  # term -> document frequency
        self.num_docs: int = 0

    # ------------------------------------------------------------------ 分词
    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词: 中文按字、英文按词, 保留顺序。

        示例:
            "地月转移用Hohmann方法" -> ['地','月','转','移','用','hohmann','方','法']
        """
        if not text:
            return []
        text = text.lower()
        tokens: List[str] = []
        cur = ""
        for ch in text:
            # CJK 统一汉字基本区
            if "\u4e00" <= ch <= "\u9fff":
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append(ch)
            elif ch.isascii() and ch.isalnum():
                cur += ch
            else:
                if cur:
                    tokens.append(cur)
                    cur = ""
        if cur:
            tokens.append(cur)
        return tokens

    # ------------------------------------------------------------------ 特征
    def _features(self, text: str) -> Dict[str, int]:
        """抽取 1/2/3-gram 特征 -> {feature: term_frequency}。

        对中文而言, 单字即 token, 因此 3-gram 天然就是「字符级 3-gram」;
        对英文, token 是整词, n-gram 为词组级特征 (配合下方对英文词内的字符
        3-gram, 兼顾子词泛化能力)。
        """
        tokens = self._tokenize(text)
        feats: Dict[str, int] = {}

        # token 序列上的 1..n-gram
        for n in range(1, self.ngram + 1):
            if n == 1:
                for t in tokens:
                    feats[t] = feats.get(t, 0) + 1
            else:
                for i in range(len(tokens) - n + 1):
                    g = "\x01".join(tokens[i : i + n])  # 用 \x01 分隔避免歧义
                    feats[g] = feats.get(g, 0) + 1

        # 英文词内字符级 3-gram (子词特征, 提升对词形变化/拼写差异的鲁棒性)
        for t in tokens:
            if len(t) >= 4 and t.isascii():  # 仅对较长的英文词做字符 n-gram
                padded = f"#{t}#"
                for i in range(len(padded) - 3 + 1):
                    cg = "c:" + padded[i : i + 3]
                    feats[cg] = feats.get(cg, 0) + 1
        return feats

    # ------------------------------------------------------------------ IDF
    def _idf(self, term: str) -> float:
        """平滑 IDF: idf(t) = ln((N+1)/(df(t)+1)) + 1。

        对语料中未出现的词, df=0, 退化为 ln(N+1)+1, 仍为有限正值。
        """
        df = self.df.get(term, 0)
        return math.log((self.num_docs + 1) / (df + 1)) + 1.0

    def fit_partial(self, text: str) -> None:
        """把一段文本计入语料统计 (更新 df / num_docs)。

        应在 :meth:`embed` 之前对每篇**文档**调用一次; 对查询文本不要调用。
        """
        for t in self._features(text):
            self.df[t] = self.df.get(t, 0) + 1
        self.num_docs += 1

    # ------------------------------------------------------------------ 嵌入
    def embed(self, text: str) -> np.ndarray:
        """把文本嵌入为 dim 维 L2 归一化向量。

        emb = Σ_term w(term) · R[hash(term), :]
        其中 w = (1 + log(tf)) · idf(term)  (sublinear TF · IDF)
        """
        feats = self._features(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for term, tf in feats.items():
            idx = _stable_hash(term) % self.vocab_size
            w = (1.0 + math.log(tf)) * self._idf(term)  # sublinear TF * IDF
            vec += w * self.proj[idx]
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    # ------------------------------------------------------------- 状态持久化
    def state(self) -> dict:
        """导出可序列化状态 (df / num_docs), 用于持久化与跨进程复现。"""
        return {"df": self.df, "num_docs": self.num_docs}

    def load_state(self, s: dict) -> None:
        self.df = dict(s.get("df", {}))
        self.num_docs = int(s.get("num_docs", 0))


class VectorStore:
    """基于余弦相似度的向量检索库。

    文档以插入顺序编号 (0,1,2,…), 与 :class:`KeywordIndex` 的 doc_id 对齐,
    便于混合检索时按 doc_id 合并结果。
    """

    def __init__(self, embedder: "SimpleEmbedder | None" = None):
        self.embedder = embedder if embedder is not None else SimpleEmbedder()
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        # 嵌入矩阵 (N × dim), 空时为 (0, dim) 形状以便后续 vstack
        self._embeddings: np.ndarray = np.zeros(
            (0, self.embedder.dim), dtype=np.float32
        )

    # ------------------------------------------------------------------ 增删
    def add(self, text: str, metadata: dict | None = None) -> int:
        """添加一篇文档, 计算 embedding 存入矩阵, 返回其 doc_id。"""
        doc_id = len(self.texts)
        self.embedder.fit_partial(text)  # 先更新语料 IDF
        vec = self.embedder.embed(text)  # 再嵌入 (此时 IDF 已含本文档)
        self.texts.append(text)
        self.metadatas.append(dict(metadata) if metadata else {})
        self._embeddings = np.vstack([self._embeddings, vec[None, :]])
        return doc_id

    def reindex(self) -> None:
        """用当前 (最终) IDF 重新嵌入所有文档, 保证整库 IDF 一致性。

        增量 add 时各文档是用「当时的」IDF 嵌入的; 批量索引完成后调用一次
        reindex 可让所有文档使用同一份最终 IDF, 提升检索一致性。
        """
        if self.texts:
            self._embeddings = np.stack(
                [self.embedder.embed(t) for t in self.texts]
            ).astype(np.float32)
        else:
            self._embeddings = np.zeros(
                (0, self.embedder.dim), dtype=np.float32
            )

    def clear(self) -> None:
        """清空所有文档与嵌入器语料统计。"""
        self.texts = []
        self.metadatas = []
        self._embeddings = np.zeros(
            (0, self.embedder.dim), dtype=np.float32
        )
        self.embedder.df = {}
        self.embedder.num_docs = 0

    def __len__(self) -> int:
        return len(self.texts)

    # ------------------------------------------------------------------ 检索
    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[float, str, dict]]:
        """余弦相似度检索。返回 [(score, text, metadata), ...] 按分数降序。

        由于向量已 L2 归一化, 余弦相似度 = 点积。
        """
        if len(self.texts) == 0:
            return []
        q = self.embedder.embed(query)  # 查询不更新语料
        # 行向量点积 = 余弦相似度 (均已归一化)
        scores = self._embeddings @ q  # (N,)
        k = min(top_k, len(self.texts))
        # argpartition 取 top_k, 再排序
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            (float(scores[i]), self.texts[i], self.metadatas[i]) for i in idx
        ]

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str) -> None:
        """保存到 path(.npz) + path 的同目录 meta.json。

        - .npz : 嵌入矩阵 + 文本数组 (numpy savez)
        - .json: 元数据 + 嵌入器 IDF 状态
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            embeddings=self._embeddings,
            texts=np.array(self.texts, dtype=object),
        )
        meta_path = path[:-4] + ".meta.json" if path.endswith(".npz") else path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadatas": self.metadatas,
                    "embedder_state": self.embedder.state(),
                    "dim": self.embedder.dim,
                    "ngram": self.embedder.ngram,
                    "vocab_size": self.embedder.vocab_size,
                    "seed": self.embedder.seed,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str) -> None:
        """从 path(.npz) + meta.json 加载。"""
        data = np.load(path, allow_pickle=True)
        self._embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        self.texts = [str(t) for t in data["texts"].tolist()]
        meta_path = path[:-4] + ".meta.json" if path.endswith(".npz") else path + ".meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.metadatas = list(meta.get("metadatas", []))
        # 恢复嵌入器状态 (注意: 投影矩阵由 seed 重新生成, 因此与保存时一致)
        self.embedder = SimpleEmbedder(
            dim=meta.get("dim", 256),
            ngram=meta.get("ngram", 3),
            vocab_size=meta.get("vocab_size", 1 << 15),
            seed=meta.get("seed", 42),
        )
        self.embedder.load_state(meta.get("embedder_state", {}))


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    vs = VectorStore()
    docs = [
        ("Hohmann 转移是共面圆轨道间最省能量的双脉冲转移", {"topic": "hohmann"}),
        ("vis-viva 方程: v^2 = mu (2/r - 1/a), 由能量守恒导出", {"topic": "vis_viva"}),
        ("地月转移通常用拼凑圆锥近似, 分地心段与月心段", {"topic": "moon_transfer"}),
        ("Lambert 问题已知两点位置与飞行时间求轨道", {"topic": "lambert"}),
        ("开普勒方程 M = E - e sinE 描述平近点角与偏近点角关系", {"topic": "kepler"}),
    ]
    for t, m in docs:
        vs.add(t, m)
    vs.reindex()

    q = "地月转移用什么方法"
    print(f"[query] {q}")
    for score, text, meta in vs.search(q, top_k=3):
        print(f"  {score:.4f}  {meta['topic']:>14}  {text}")

    # 可复现性测试: 相同文本 -> 相同向量
    e1 = vs.embedder.embed("Hohmann 转移")
    e2 = vs.embedder.embed("Hohmann 转移")
    assert np.allclose(e1, e2), "可复现性失败"
    print("[ok] 可复现性: 相同文本向量一致")

    # 持久化测试
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vs.npz")
        vs.save(p)
        vs2 = VectorStore()
        vs2.load(p)
        r1 = vs.search(q, top_k=3)
        r2 = vs2.search(q, top_k=3)
        same = all(
            np.allclose(a[0], b[0]) and a[1] == b[1] for a, b in zip(r1, r2)
        )
        print(f"[ok] 持久化: 保存/加载后检索结果一致 = {same}")
