"""记忆系统。

- ``ShortTermMemory`` : 最近 N 轮对话（deque 自动淘汰）
- ``LongTermMemory``  : JSON 文件持久化 + 简化向量检索
  （用 numpy 点积 + 随机投影做 embedding 占位，无需安装向量库/tokenizer）

简化 embedding 原理：将文本转为字符袋（bag-of-chars）向量，再用固定的随机
投影矩阵降维到 embed_dim，归一化后用 numpy 点积计算余弦相似度。该实现稳定、
无外部依赖，足以演示检索流程；真实场景可替换为预训练 embedding。
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ShortTermMemory:
    """短期记忆：最近 N 轮对话，使用 deque 自动淘汰旧记录。"""

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds
        self.rounds: deque = deque(maxlen=max_rounds)

    def add(self, role: str, content: str) -> None:
        """添加一条消息。"""
        self.rounds.append({"role": role, "content": content})

    def add_round(self, user: str, assistant: str) -> None:
        """添加一轮完整的 user + assistant 对。"""
        self.rounds.append({"role": "user", "content": user})
        self.rounds.append({"role": "assistant", "content": assistant})

    def get_all(self) -> List[Dict[str, str]]:
        """返回全部消息列表。"""
        return list(self.rounds)

    def to_messages(self) -> List[Dict[str, str]]:
        """转为 LLM 消息格式。"""
        return list(self.rounds)

    def clear(self) -> None:
        """清空短期记忆。"""
        self.rounds.clear()

    def __len__(self) -> int:
        return len(self.rounds)


class LongTermMemory:
    """长期记忆：JSON 文件持久化 + 简化向量检索。"""

    DEFAULT_PATH = "/workspace/data/memory.json"

    def __init__(self, path: str = None, embed_dim: int = 64,
                 proj_seed: int = 42):
        """
        Args:
            path: 持久化 JSON 文件路径
            embed_dim: embedding 输出维度
            proj_seed: 随机投影矩阵种子（固定以保证跨运行一致性）
        """
        self.path = Path(path or self.DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.proj_seed = proj_seed
        # 输入维度覆盖常见字符（含中文按 ord 取模）
        self.input_dim = 256
        # 固定随机投影矩阵 (input_dim x embed_dim)
        rng = np.random.RandomState(self.proj_seed)
        self._proj = rng.randn(self.input_dim, self.embed_dim).astype(np.float32)
        # 键值存储：key -> {value, tags, embedding}
        self.store: Dict[str, Dict[str, Any]] = {}
        self.load()

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------
    def _bag_of_chars(self, text: str) -> np.ndarray:
        """将文本转为字符袋向量（维度=input_dim）。"""
        vec = np.zeros(self.input_dim, dtype=np.float32)
        for ch in text:
            vec[ord(ch) % self.input_dim] += 1.0
        return vec

    def embed(self, text: str) -> np.ndarray:
        """对文本做随机投影 embedding 并归一化。"""
        boc = self._bag_of_chars(text)
        emb = boc @ self._proj  # (embed_dim,)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def remember(self, key: str, value: Any, tags: List[str] = None) -> None:
        """写入一条长期记忆。

        Args:
            key: 记忆键名
            value: 记忆内容（字符串或可 JSON 序列化对象）
            tags: 标签列表，便于分类
        """
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False
        )
        emb = self.embed(f"{key} {text}")
        self.store[key] = {
            "value": value,
            "tags": tags or [],
            "embedding": emb.tolist(),
        }

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Any]]:
        """检索与 query 最相关的 top_k 条记忆。

        Returns:
            列表，元素为 (key, similarity, value)，按相似度降序
        """
        if not self.store:
            return []
        q_emb = self.embed(query)
        results: List[Tuple[str, float, Any]] = []
        for key, item in self.store.items():
            emb = np.array(item["embedding"], dtype=np.float32)
            sim = float(np.dot(q_emb, emb))
            results.append((key, sim, item["value"]))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get(self, key: str) -> Optional[Any]:
        """按键取值。"""
        return self.store.get(key, {}).get("value")

    def has(self, key: str) -> bool:
        return key in self.store

    def keys(self) -> List[str]:
        return list(self.store.keys())

    def items(self) -> List[Tuple[str, Any, List[str]]]:
        """返回全部 (key, value, tags)。"""
        return [(k, v["value"], v["tags"]) for k, v in self.store.items()]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self) -> None:
        """持久化到 JSON 文件。"""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"store": self.store}, f, ensure_ascii=False, default=str)

    def load(self) -> None:
        """从 JSON 文件加载（文件不存在或损坏则空初始化）。"""
        if not self.path.exists():
            self.store = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.store = data.get("store", {})
        except (json.JSONDecodeError, OSError):
            self.store = {}

    def __len__(self) -> int:
        return len(self.store)
