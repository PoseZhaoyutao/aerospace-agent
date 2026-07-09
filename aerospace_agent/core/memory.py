"""记忆系统。

三层记忆架构：
    - ``ShortTermMemory``  : 最近 N 轮对话（deque 自动淘汰）
    - ``WorkingMemory``   : 当前任务的工作记忆（任务状态、中间变量、临时结论）
    - ``LongTermMemory``   : JSON 文件持久化 + 简化向量检索

``MemoryManager`` 统一管理三层记忆，提供：
    - 初始化记忆（预置航天领域基础知识）
    - 统一写入/检索接口
    - 跨层协作（短期→工作→长期 的自动晋升）

简化 embedding 原理：将文本转为字符袋（bag-of-chars）向量，再用固定的随机
投影矩阵降维到 embed_dim，归一化后用 numpy 点积计算余弦相似度。该实现稳定、
无外部依赖，足以演示检索流程；真实场景可替换为预训练 embedding。
"""
from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _default_data_dir() -> str:
    """获取默认数据目录（Windows 兼容）。"""
    return os.environ.get("AEROSPACE_DATA_DIR",
                          os.path.join(os.getcwd(), "data"))


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


class WorkingMemory:
    """工作记忆：当前任务的临时状态、中间变量、临时结论。

    介于短期记忆（对话历史）和长期记忆（持久化）之间：
      - 任务执行期间活跃，任务完成后可选晋升到长期记忆
      - 存储中间计算结果、任务进度、临时假设
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._task_context: Dict[str, Any] = {}
        self._created_at: str = datetime.now().isoformat()

    def set(self, key: str, value: Any) -> None:
        """设置工作记忆项。"""
        self._store[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取工作记忆项。"""
        return self._store.get(key, {}).get("value", default)

    def has(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        """删除工作记忆项。"""
        self._store.pop(key, None)

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def set_task_context(self, key: str, value: Any) -> None:
        """设置任务上下文（任务目标、约束、当前阶段等）。"""
        self._task_context[key] = value

    def get_task_context(self, key: str = None) -> Any:
        """获取任务上下文。"""
        if key is None:
            return dict(self._task_context)
        return self._task_context.get(key)

    def clear(self) -> None:
        """清空工作记忆（保留任务上下文）。"""
        self._store.clear()

    def all_items(self) -> Dict[str, Any]:
        """返回所有工作记忆项。"""
        return {k: v["value"] for k, v in self._store.items()}

    def __len__(self) -> int:
        return len(self._store)


class LongTermMemory:
    """长期记忆：JSON 文件持久化 + 简化向量检索。"""

    def __init__(self, path: str = None, embed_dim: int = 64,
                 proj_seed: int = 42):
        """
        Args:
            path: 持久化 JSON 文件路径（默认 data/memory.json）
            embed_dim: embedding 输出维度
            proj_seed: 随机投影矩阵种子（固定以保证跨运行一致性）
        """
        default_path = os.path.join(_default_data_dir(), "memory.json")
        self.path = Path(path or default_path)
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
        """从 JSON 文件加载（文件不存在或损坏则空初始化）。

        K5-缺陷14: 校验已存 embedding 维度，不匹配则丢弃避免后续 np.dot 崩溃。
        """
        if not self.path.exists():
            self.store = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_store = data.get("store", {})
            if not isinstance(loaded_store, dict):
                loaded_store = {}
            # K5-缺陷14: 校验 embedding 维度
            valid_store = {}
            for key, item in loaded_store.items():
                if not isinstance(item, dict):
                    continue
                emb = item.get("embedding")
                if emb is not None and len(emb) != self.embed_dim:
                    # 维度不匹配，重新计算 embedding
                    text = item.get("value", "")
                    item["embedding"] = self.embed(text).tolist()
                valid_store[key] = item
            self.store = valid_store
        except (json.JSONDecodeError, OSError):
            self.store = {}

    def __len__(self) -> int:
        return len(self.store)


# ======================================================================
# MemoryManager —— 三层记忆统一管理器
# ======================================================================

def _default_aerospace_knowledge() -> List[Dict[str, Any]]:
    """预置航天领域基础知识条目，用于初始化长期记忆。

    Returns:
        列表，每项为 {"key": str, "value": str, "tags": List[str]}
    """
    return [
        {
            "key": "earth_mu",
            "value": "地球引力常数 mu = 398600.4418 km^3/s^2 (GM)，"
                     "用于所有地球轨道计算。",
            "tags": ["常数", "地球", "轨道力学"],
        },
        {
            "key": "leo_velocity",
            "value": "低地球轨道(LEO, h≈400km)圆轨道速度约 7.67 km/s，"
                     "轨道周期约 92.7 分钟。",
            "tags": ["LEO", "轨道速度", "基础"],
        },
        {
            "key": "geo_altitude",
            "value": "地球静止轨道(GEO)高度约 35786 km，轨道周期等于"
                     "一个恒星日(约 23h56m4s)，轨道速度约 3.07 km/s。",
            "tags": ["GEO", "静止轨道", "基础"],
        },
        {
            "key": "escape_velocity",
            "value": "地球表面逃逸速度约 11.19 km/s；近地轨道逃逸速度"
                     "约 10.9 km/s。",
            "tags": ["逃逸速度", "基础"],
        },
        {
            "key": "tli_delta_v",
            "value": "地月转移轨道(TLI)所需速度增量约 3.05-3.15 km/s，"
                     "从 LEO(200km)出发。地月转移时间约 3-5 天。",
            "tags": ["地月转移", "TLI", "轨道机动"],
        },
        {
            "key": "lambert_problem",
            "value": "Lambert 问题：已知两个位置点和飞行时间，求解连接"
                     "两点的轨道。是轨道交会与转移轨道设计的基础。",
            "tags": ["Lambert", "轨道设计", "基础概念"],
        },
        {
            "key": "patched_conic",
            "value": "拼凑圆锥近似法：将行星际轨道分解为若干二体弧段，"
                     "在影响球边界拼接。用于初步轨道设计。",
            "tags": ["拼凑圆锥", "行星际", "近似方法"],
        },
        {
            "key": "j2_perturbation",
            "value": "J2 摄动：地球扁率引起的轨道进动。RAAN 进动率"
                     "约 -3/2 * J2 * (Re/a)^2 * n * cos(i)。"
                     "太阳同步轨道利用 J2 进动维持恒定光照角。",
            "tags": ["J2", "摄动", "轨道力学"],
        },
        {
            "key": "hohmann_transfer",
            "value": "霍曼转移：两共面圆轨道间最省燃料的双脉冲转移。"
                     "总 Delta-V = |Δv1| + |Δv2|。",
            "tags": ["霍曼转移", "轨道机动", "基础"],
        },
        {
            "key": "canonical_units",
            "value": "Canonical Astrodynamics Model 使用 SI 单位："
                     "距离(m)、速度(m/s)、时间(s)、角度(rad)。"
                     "所有引擎 I/O 通过 Canonical Model 转换，"
                     "保证无损往返。",
            "tags": ["Canonical Model", "单位制", "架构"],
        },
        {
            "key": "loop_eight_phases",
            "value": "Loop 八阶段：Plan(递归第一性原理) → "
                     "SelectEngine → RetrieveDemo → GenerateWorkflow → "
                     "Run → Validate → Fix → Save。"
                     "LoopLedger 记录每阶段状态。",
            "tags": ["Loop", "工作流", "架构"],
        },
        {
            "key": "react_pattern",
            "value": "ReAct 模式：Thought → Action → Observation 循环。"
                     "Agent 在每一步先推理(Thought)，再选择工具(Action)，"
                     "观察结果(Observation)后继续推理，直到完成任务。",
            "tags": ["ReAct", "Agent", "架构"],
        },
    ]


class MemoryManager:
    """三层记忆统一管理器。

    统一管理 ShortTermMemory + WorkingMemory + LongTermMemory，提供：
      - 统一写入/检索接口
      - 跨层协作（短期→工作→长期 的自动晋升）
      - 初始化记忆（预置航天领域基础知识）
      - 任务生命周期管理（开始任务、结束任务、晋升记忆）

    用法::

        manager = MemoryManager()
        manager.initialize()                # 预置航天知识
        manager.start_task("地月转移轨道设计")  # 开始任务
        manager.add_conversation("user", "设计一条TLI轨道")
        manager.set_working("current_phase", "plan")
        # ... 任务执行中 ...
        manager.promote_to_long_term("tli_result", "Delta-V=3.1km/s")
        manager.end_task()                  # 结束任务，保存
    """

    def __init__(
        self,
        short_term_capacity: int = 20,
        long_term_path: str = None,
        auto_initialize: bool = True,
    ):
        """
        Args:
            short_term_capacity: 短期记忆最大轮次
            long_term_path: 长期记忆文件路径
            auto_initialize: 是否自动初始化预置知识
        """
        self.short_term = ShortTermMemory(max_rounds=short_term_capacity)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(path=long_term_path)
        self._initialized = False
        if auto_initialize:
            self.initialize()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initialize(self, force: bool = False) -> int:
        """初始化长期记忆，预置航天领域基础知识。

        Args:
            force: True 则覆盖已有同名条目；False 则跳过已有条目

        Returns:
            新增的条目数
        """
        if self._initialized and not force:
            return 0
        count = 0
        for item in _default_aerospace_knowledge():
            if force or not self.long_term.has(item["key"]):
                self.long_term.remember(
                    item["key"], item["value"], item["tags"]
                )
                count += 1
        self.long_term.save()
        self._initialized = True
        return count

    # ------------------------------------------------------------------
    # 短期记忆接口
    # ------------------------------------------------------------------
    def add_conversation(self, role: str, content: str) -> None:
        """添加一条对话消息到短期记忆。"""
        self.short_term.add(role, content)

    def add_round(self, user: str, assistant: str) -> None:
        """添加一轮完整的对话。"""
        self.short_term.add_round(user, assistant)

    def get_recent_messages(self) -> List[Dict[str, str]]:
        """获取短期记忆中的消息列表（LLM 格式）。"""
        return self.short_term.to_messages()

    # ------------------------------------------------------------------
    # 工作记忆接口
    # ------------------------------------------------------------------
    def set_working(self, key: str, value: Any) -> None:
        """设置工作记忆项。"""
        self.working.set(key, value)

    def get_working(self, key: str, default: Any = None) -> Any:
        """获取工作记忆项。"""
        return self.working.get(key, default)

    def set_task_context(self, key: str, value: Any) -> None:
        """设置任务上下文。"""
        self.working.set_task_context(key, value)

    def get_task_context(self, key: str = None) -> Any:
        """获取任务上下文。"""
        return self.working.get_task_context(key)

    # ------------------------------------------------------------------
    # 长期记忆接口
    # ------------------------------------------------------------------
    def remember(self, key: str, value: Any, tags: List[str] = None) -> None:
        """写入长期记忆。"""
        self.long_term.remember(key, value, tags)
        self.long_term.save()

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Any]]:
        """检索长期记忆。"""
        return self.long_term.recall(query, top_k)

    def get_memory(self, key: str) -> Optional[Any]:
        """按键从长期记忆取值。"""
        return self.long_term.get(key)

    # ------------------------------------------------------------------
    # 跨层协作
    # ------------------------------------------------------------------
    def promote_to_long_term(
        self, key: str, value: Any, tags: List[str] = None
    ) -> None:
        """将工作记忆项晋升到长期记忆。

        通常在任务完成或发现重要结论时调用。
        """
        self.long_term.remember(key, value, tags)
        self.long_term.save()

    def recall_to_working(self, query: str, top_k: int = 3) -> List[str]:
        """从长期记忆检索相关条目，注入工作记忆。

        Returns:
            检索到的记忆 key 列表
        """
        results = self.long_term.recall(query, top_k)
        keys = []
        for key, sim, value in results:
            wm_key = f"recall_{key}"
            self.working.set(wm_key, {"value": value, "similarity": sim})
            keys.append(key)
        return keys

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------
    def start_task(self, task_description: str) -> None:
        """开始一个新任务。

        清空工作记忆，设置任务上下文，记录任务开始时间。
        """
        self.working.clear()
        self.working.set_task_context("task", task_description)
        self.working.set_task_context("start_time", datetime.now().isoformat())
        self.working.set_task_context("phase", "started")

    def end_task(self, save: bool = True) -> None:
        """结束当前任务。

        Args:
            save: 是否保存长期记忆
        """
        self.working.set_task_context("end_time", datetime.now().isoformat())
        self.working.set_task_context("phase", "completed")
        if save:
            self.long_term.save()

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """返回三层记忆的完整状态。"""
        return {
            "short_term": {
                "messages": len(self.short_term),
                "capacity": self.short_term.max_rounds,
            },
            "working": {
                "items": len(self.working),
                "task_context": self.working.get_task_context(),
            },
            "long_term": {
                "items": len(self.long_term),
                "path": str(self.long_term.path),
            },
            "initialized": self._initialized,
        }

    def save(self) -> None:
        """保存长期记忆到磁盘。"""
        self.long_term.save()

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"MemoryManager(short={s['short_term']['messages']}, "
            f"working={s['working']['items']}, "
            f"long={s['long_term']['items']})"
        )
