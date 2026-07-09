"""WorkflowCatalog — 工作流目录加载与检索。

第一性原理：Loop 的 GenerateWorkflow 阶段需要一个可检索的工作流库，
使 LLM 能基于 query/task_type/engine 复用已验证的 WorkflowSpec，
而非每次从零生成。

目录分为 7 大类别（CATALOG_CATEGORIES），覆盖全部任务类型。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from ..schemas import WorkflowSpec


#: 7 大工作流类别——对应 task_type
CATALOG_CATEGORIES: List[str] = [
    "orbit_propagation",   # 轨道传播
    "frame_transform",     # 坐标系转换
    "ephemeris",           # 星历查询
    "ground_access",       # 地面站可见性
    "maneuver",            # 轨道机动
    "attitude_control",    # 姿态控制
    "validation",          # 交叉验证
]


class WorkflowCatalog:
    """工作流目录——加载、检索、管理可复用 WorkflowSpec。

    典型用法::

        catalog = WorkflowCatalog()
        catalog.load_from_dir(".../workflows")
        results = catalog.search(query="LEO", task_type="orbit_propagation")

    约束：只读加载源 YAML，不修改源文件。
    """

    def __init__(self) -> None:
        self._workflows: Dict[str, dict] = {}
        # 按类别分组：task_type -> [workflow_id, ...]
        self._by_category: Dict[str, List[str]] = {c: [] for c in CATALOG_CATEGORIES}

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load_from_dir(self, yaml_dir: Union[str, Path]) -> int:
        """从目录加载所有 ``.yaml`` 工作流文件。

        Args:
            yaml_dir: 包含 .yaml 工作流文件的目录

        Returns:
            成功加载的工作流数量

        单个文件解析失败不阻断整体加载（跳过并继续）。
        """
        yaml_dir = Path(yaml_dir)
        if not yaml_dir.is_dir():
            return 0
        count = 0
        for fp in sorted(yaml_dir.glob("*.yaml")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                wf = WorkflowSpec.from_yaml_dict(data).to_dict()
                wf_id = wf.get("id") or fp.stem
                wf["id"] = wf_id
                # 记录源文件位置（便于追溯，不修改源文件）
                wf.setdefault("metadata", {})["source_file"] = str(fp)
                self._workflows[wf_id] = wf
                # 按 task_type 分类
                cat = wf.get("task_type", "") or "uncategorized"
                bucket = self._by_category.setdefault(cat, [])
                if wf_id not in bucket:
                    bucket.append(wf_id)
                count += 1
            except Exception:
                # 单个文件解析失败，跳过不中断
                continue
        return count

    def load_from_dict(self, data: dict) -> bool:
        """从字典直接加载单个工作流。

        Returns:
            是否加载成功
        """
        if not isinstance(data, dict):
            return False
        try:
            wf = WorkflowSpec.from_yaml_dict(data).to_dict()
            wf_id = wf.get("id") or ""
            if not wf_id:
                return False
            self._workflows[wf_id] = wf
            cat = wf.get("task_type", "") or "uncategorized"
            bucket = self._by_category.setdefault(cat, [])
            if wf_id not in bucket:
                bucket.append(wf_id)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def get(self, workflow_id: str) -> Optional[dict]:
        """按 ID 获取完整工作流字典。"""
        return self._workflows.get(workflow_id)

    def list_all(self) -> List[dict]:
        """列出所有工作流的元数据摘要。"""
        result = []
        for wf_id, wf in self._workflows.items():
            result.append({
                "id": wf_id,
                "goal": wf.get("goal", ""),
                "task_type": wf.get("task_type", ""),
                "engine": wf.get("engine", "auto"),
                "steps_count": len(wf.get("steps", [])),
            })
        return result

    def search(self, query: str = "", task_type: Optional[str] = None,
               preferred_engine: Optional[str] = None) -> List[dict]:
        """按 query/task_type/engine 检索工作流。

        Args:
            query: 关键词（匹配 goal/id/task_type/步骤 tool/name/description）
            task_type: 任务类别（CATALOG_CATEGORIES 之一）
            preferred_engine: 偏好引擎；``auto`` 引擎的工作流始终匹配

        Returns:
            匹配的工作流列表
        """
        results: List[dict] = []
        q = query.lower().strip() if query else ""
        for wf in self._workflows.values():
            # task_type 过滤
            if task_type is not None and wf.get("task_type", "") != task_type:
                continue
            # engine 过滤（auto 视为匹配任意偏好引擎）
            if preferred_engine is not None:
                wf_engine = wf.get("engine", "auto")
                if wf_engine not in ("auto", preferred_engine):
                    continue
            # 关键词匹配
            if q:
                haystack_parts = [
                    str(wf.get("id", "")).lower(),
                    str(wf.get("goal", "")).lower(),
                    str(wf.get("task_type", "")).lower(),
                ]
                for step in wf.get("steps", []):
                    haystack_parts.append(str(step.get("tool", "")).lower())
                    haystack_parts.append(str(step.get("name", "")).lower())
                    haystack_parts.append(str(step.get("description", "")).lower())
                haystack = " ".join(haystack_parts)
                if q not in haystack:
                    continue
            results.append(wf)
        return results

    def categories(self) -> Dict[str, List[str]]:
        """返回类别 → 工作流 ID 列表映射。"""
        return {c: list(ids) for c, ids in self._by_category.items()}

    def ids(self) -> List[str]:
        """返回所有工作流 ID。"""
        return list(self._workflows.keys())

    def __len__(self) -> int:
        return len(self._workflows)

    def __contains__(self, workflow_id: object) -> bool:
        return workflow_id in self._workflows
