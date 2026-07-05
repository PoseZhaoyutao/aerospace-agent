"""CEO 上下文管理策略。

CEO = Compress / Essential-preserve / Offload。

核心硬性要求：**目标任务设计（任务规格、关键公式、用户原始指令）不被压缩失真**，
始终在 Essential 层原样保留，无论 token 预算多紧张都不会被截断或摘要。

三层结构：
    - Essential 层：任务规格、关键公式、用户原始指令——永不压缩，原样保留
    - Compress  层：中间对话历史、工具调用记录——超阈值时摘要压缩
    - Offload   层：大块数据、检索结果、轨迹数据——存外部文件，上下文只保留引用
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ContextManager:
    """CEO 三层上下文管理器。"""

    def __init__(self, offload_dir: str = "/workspace/data/offload",
                 compress_threshold: int = 2000,
                 keep_recent: int = 6):
        """
        Args:
            offload_dir: Offload 层数据的存储目录
            compress_threshold: Compress 层触发压缩的 token 阈值
            keep_recent: 压缩时保留最近 N 条原文
        """
        self.offload_dir = Path(offload_dir)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self.compress_threshold = compress_threshold
        self.keep_recent = keep_recent

        # Essential 层：永不压缩，原样保留
        self.essential_items: List[str] = []
        # Compress 层：对话历史 + 工具调用记录
        self.messages: List[Dict[str, str]] = []
        self.tool_records: List[Dict[str, Any]] = []
        # Offload 层索引：key -> {path, summary, size}
        self.offload_index: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Essential 层
    # ------------------------------------------------------------------
    def add_essential(self, text: str) -> None:
        """添加 Essential 层内容。

        该内容将原样保留，永不压缩或截断（任务规格、关键公式、用户原始指令等）。
        """
        if text and text not in self.essential_items:
            self.essential_items.append(text)

    # ------------------------------------------------------------------
    # Compress 层
    # ------------------------------------------------------------------
    def add_message(self, role: str, content: str) -> None:
        """添加一条对话消息到 Compress 层。"""
        self.messages.append({"role": role, "content": content})

    def add_tool_record(self, tool: str, args: Any, result: Any) -> None:
        """添加一条工具调用记录到 Compress 层。"""
        self.tool_records.append({"tool": tool, "args": args, "result": result})

    # ------------------------------------------------------------------
    # Offload 层
    # ------------------------------------------------------------------
    def save_offload(self, key: str, data: Any) -> str:
        """将大块数据保存到外部文件，上下文只保留引用。

        Args:
            key: 数据键名（用于检索）
            data: 任意可 JSON 序列化的数据

        Returns:
            数据文件的绝对路径
        """
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        path = self.offload_dir / f"{digest}.json"
        payload = {"key": key, "data": data}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
        summary = self._summarize_data(data)
        self.offload_index[key] = {
            "path": str(path),
            "summary": summary,
            "size": path.stat().st_size,
        }
        return str(path)

    def load_offload(self, key: str) -> Optional[Any]:
        """根据 key 从外部文件加载 Offload 层数据。"""
        info = self.offload_index.get(key)
        if not info:
            return None
        path = Path(info["path"])
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("data")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """简单 token 估算：字符数 / 3.5（无需安装 tokenizer）。"""
        if not text:
            return 0
        return max(1, int(len(text) / 3.5))

    def _to_text(self, obj: Any) -> str:
        """将任意对象转为文本。"""
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    def _summarize_data(self, data: Any) -> str:
        """生成数据的简短摘要（用于上下文引用）。"""
        text = self._to_text(data)
        if len(text) <= 80:
            return text
        return text[:80] + "...[已截断]"

    def _compress_messages(self) -> List[str]:
        """压缩 Compress 层消息：超阈值时对旧消息摘要，保留最近若干条原文。"""
        rendered = [f"[{m['role']}] {m['content']}" for m in self.messages]
        total = sum(self._estimate_tokens(r) for r in rendered)
        if total <= self.compress_threshold or len(rendered) <= self.keep_recent:
            return rendered
        tail = rendered[-self.keep_recent:]
        head = rendered[:-self.keep_recent]
        # 对旧消息做简单摘要：保留角色标记与首 60 字
        compressed_head = []
        for r in head:
            if "]" in r:
                tag, body = r.split("]", 1)
                tag = tag + "]"
            else:
                tag, body = "", r
            compressed_head.append(f"{tag} {body.strip()[:60]}...[已压缩]")
        return compressed_head + tail

    def _compress_tool_records(self) -> List[str]:
        """压缩工具调用记录：旧记录摘要，保留最近若干条。"""
        if not self.tool_records:
            return []
        rendered = [
            f"[tool:{rec['tool']}] args={self._summarize_data(rec['args'])} "
            f"-> {self._to_text(rec['result'])[:120]}"
            for rec in self.tool_records
        ]
        if len(rendered) <= self.keep_recent:
            return rendered
        tail = rendered[-self.keep_recent:]
        head = rendered[:-self.keep_recent]
        compressed_head = [f"{r[:80]}...[已压缩]" for r in head]
        return compressed_head + tail

    # ------------------------------------------------------------------
    # 构建上下文
    # ------------------------------------------------------------------
    def build_context(self, token_budget: int = 8000) -> str:
        """构建最终上下文字符串。

        策略（保证 Essential 层原样保留的硬性要求）：
          1. Essential 层永远完整保留——即使超出预算也不压缩/截断；
          2. Offload 层只放入引用（key + 摘要 + 路径），不放入原始数据；
          3. Compress 层（消息 + 工具记录）先按阈值压缩，再按剩余预算
             从最新到最旧保留，超出部分截断（仅截断 Compress 层，绝不截断 Essential）。
        """
        sections: List[str] = []

        # 1. Essential 层（原样保留，永不压缩）
        if self.essential_items:
            sections.append("===== Essential（任务规格，原样保留，永不压缩）=====")
            sections.extend(self.essential_items)

        # 2. Offload 层引用
        if self.offload_index:
            sections.append("===== Offload（外部数据引用）=====")
            for key, info in self.offload_index.items():
                sections.append(
                    f"- [{key}] -> {info['path']} | 摘要: {info['summary']} "
                    f"| {info['size']} 字节"
                )

        # 计算已用 token（Essential + Offload 引用）
        used = sum(self._estimate_tokens(s) for s in sections)
        remaining = max(0, token_budget - used)

        # 3. Compress 层（先压缩，再按剩余预算从最新到最旧保留）
        compress_block: List[str] = []
        compressed_tools = self._compress_tool_records()
        compressed_msgs = self._compress_messages()
        if compressed_tools:
            compress_block.append("----- 工具调用记录 -----")
            compress_block.extend(compressed_tools)
        if compressed_msgs:
            compress_block.append("----- 对话历史 -----")
            compress_block.extend(compressed_msgs)

        kept: List[str] = []
        used_compress = 0
        for line in reversed(compress_block):
            t = self._estimate_tokens(line)
            if kept and used_compress + t > remaining:
                break
            kept.append(line)
            used_compress += t
        kept.reverse()

        if kept:
            sections.append("===== Compress（历史/工具，超阈值已压缩）=====")
            sections.extend(kept)

        return "\n".join(sections)

    def clear_compressed(self) -> None:
        """清空 Compress 层（保留 Essential 与 Offload 索引）。"""
        self.messages.clear()
        self.tool_records.clear()

    def stats(self) -> Dict[str, int]:
        """返回各层条目统计，便于调试。"""
        return {
            "essential": len(self.essential_items),
            "messages": len(self.messages),
            "tool_records": len(self.tool_records),
            "offload": len(self.offload_index),
        }
