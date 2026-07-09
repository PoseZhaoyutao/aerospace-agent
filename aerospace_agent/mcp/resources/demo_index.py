"""DemoIndex — 示例索引与检索。

第一性原理：Loop 的 RetrieveDemo 阶段需要一个可检索的示例库，
使 LLM 能基于相似任务检索历史成功工作流/请求作为 few-shot 参考，
降低从零生成的出错率。

重要约束：本类只读扫描源目录，**绝不修改源目录内容**。
索引可持久化为 JSON 文件供下次会话复用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DemoIndex:
    """Demo 索引——扫描示例目录，提取元数据，支持检索。

    典型用法::

        idx = DemoIndex(index_path=".../demo_index.json")
        idx.index_demos(scan_paths=[".../workflows", ".../examples"])
        hits = idx.search_demos(query="地面站", task_type="ground_access")

    约束：只读扫描源目录，不写入源目录；索引可另存为 JSON。
    """

    def __init__(self, index_path: Optional[Union[str, Path]] = None) -> None:
        self._index: Dict[str, dict] = {}
        self._index_path: Optional[Path] = (
            Path(index_path) if index_path else None
        )

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def index_demos(
        self,
        sources: Optional[Dict[str, str]] = None,
        scan_paths: Optional[List[Union[str, Path]]] = None,
    ) -> dict:
        """扫描路径中的示例，提取元数据构建索引。

        Args:
            sources: 源标签 → 路径映射（如 ``{"workflows": ".../workflows"}``）
            scan_paths: 额外扫描路径列表

        Returns:
            索引摘要 ``{"total": N, "by_type": {task_type: count}}``

        约束：只读扫描，绝不修改源目录。
        """
        sources = sources or {}
        scan_paths = list(scan_paths or [])
        # 合并所有待扫描路径
        all_paths: List[Path] = []
        for p in sources.values():
            all_paths.append(Path(p))
        for p in scan_paths:
            all_paths.append(Path(p))

        for base in all_paths:
            if not base.exists():
                continue
            if base.is_file():
                self._index_file(base)
            else:
                for fp in sorted(base.rglob("*")):
                    if fp.is_file():
                        self._index_file(fp)

        summary = self._summary()
        # 可选持久化到 JSON（写入索引文件，不写入源目录）
        if self._index_path is not None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        return summary

    def _index_file(self, fp: Path) -> None:
        """索引单个文件（提取元数据，不修改文件）。"""
        suffix = fp.suffix.lower()
        meta: Dict[str, Any] = {
            "path": str(fp),
            "name": fp.name,
            "suffix": suffix,
            "size_bytes": fp.stat().st_size if fp.exists() else 0,
            "id": fp.stem,
            "task_type": "",
            "engine": "auto",
        }
        try:
            if suffix in (".yaml", ".yml"):
                import yaml
                with open(fp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    meta["task_type"] = data.get("task_type", "")
                    meta["goal"] = data.get("goal", "")
                    meta["engine"] = data.get("engine", "auto")
                    meta["id"] = data.get("id", fp.stem)
                    meta["tools"] = [
                        s.get("tool", "")
                        for s in data.get("steps", [])
                        if isinstance(s, dict)
                    ]
            elif suffix == ".json":
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    meta["mission"] = data.get("mission", "")
                    meta["task_type"] = data.get("task_type", "")
                    meta["id"] = data.get("id", fp.stem)
        except Exception:
            # 解析失败保留基础元数据
            pass
        self._index[str(fp)] = meta

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search_demos(
        self,
        query: str = "",
        engine: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[dict]:
        """检索示例。

        Args:
            query: 关键词（匹配所有元数据字段）
            engine: 过滤引擎；``auto`` 引擎的示例始终匹配
            task_type: 过滤任务类别

        Returns:
            匹配的示例元数据列表
        """
        q = query.lower().strip() if query else ""
        results: List[dict] = []
        for meta in self._index.values():
            # engine 过滤（auto 视为匹配任意偏好引擎）
            if engine is not None:
                me = meta.get("engine", "auto")
                if me not in ("auto", engine):
                    continue
            # task_type 过滤
            if task_type is not None and meta.get("task_type", "") != task_type:
                continue
            # 关键词匹配
            if q:
                haystack = " ".join(str(v) for v in meta.values()).lower()
                if q not in haystack:
                    continue
            results.append(meta)
        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """将索引保存为 JSON 文件。"""
        target = Path(path) if path else self._index_path
        if target is None:
            raise ValueError("未指定索引保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def load(self, path: Optional[Union[str, Path]] = None) -> int:
        """从 JSON 文件加载已有索引。

        Returns:
            加载的条目数
        """
        target = Path(path) if path else self._index_path
        if target is None or not target.exists():
            return 0
        with open(target, "r", encoding="utf-8") as f:
            self._index = json.load(f)
        return len(self._index)

    def _summary(self) -> dict:
        """生成索引摘要。"""
        by_type: Dict[str, int] = {}
        for meta in self._index.values():
            t = meta.get("task_type", "") or "unknown"
            by_type[t] = by_type.get(t, 0) + 1
        return {"total": len(self._index), "by_type": by_type}

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        return key in self._index
