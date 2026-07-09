"""技能注册表 —— 技能的注册、检索、执行与自动发现。

SkillRegistry 是 Agent 与技能之间的中介层：
    - register/unregister: 动态增删技能
    - get/list_skills: 按名称或分类检索技能
    - execute: 通过名称调用技能的 execute 方法
    - auto_discover: 扫描 skills/ 目录自动注册所有内置技能
"""
from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .base import SkillBase
from .manifest import discover_skill_manifests


class SkillRegistry:
    """技能注册表。

    管理所有已注册的 SkillBase 实例，提供按名称/分类的检索与统一执行入口。

    Usage::

        registry = SkillRegistry()
        registry.auto_discover()
        result = registry.execute(agent, "memory_recall", query="地月转移")
    """

    def __init__(self, skill_roots: Optional[Iterable[str | Path]] = None):
        """初始化空注册表。"""
        self._skills: Dict[str, SkillBase] = {}
        self.skill_roots = [Path(root) for root in (skill_roots or [])]
        self._manifests: Dict[str, dict] = {}

    def register(self, skill: SkillBase) -> None:
        """注册一个技能实例。

        Args:
            skill: SkillBase 子类实例

        Raises:
            ValueError: skill 不是 SkillBase 实例或 name 为空
        """
        if not isinstance(skill, SkillBase):
            raise ValueError(f"注册对象必须是 SkillBase 实例，得到 {type(skill)}")
        if not skill.name:
            raise ValueError("技能 name 不能为空")
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """按名称注销技能（不存在则静默忽略）。"""
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[SkillBase]:
        """按名称获取技能实例，不存在返回 None。"""
        return self._skills.get(name)

    def list_skills(self, category: str = None) -> List[dict]:
        """列出已注册技能的元数据。

        Args:
            category: 可选分类过滤（如 "memory"/"rag"），None 表示全部

        Returns:
            技能元数据字典列表（每个含 name/description/category/available）
        """
        result: List[dict] = []
        for skill in self._skills.values():
            if category and skill.category != category:
                continue
            result.append(skill.info())
        return result

    def discover_manifests(self, roots: Optional[Iterable[str | Path]] = None) -> int:
        """Discover declarative ``SKILL.md`` manifests without executing them."""
        scan_roots = [Path(root) for root in (roots or self.skill_roots)]
        count = 0
        for manifest in discover_skill_manifests(scan_roots):
            name = manifest.get("name")
            if not name:
                continue
            if name not in self._manifests:
                count += 1
            self._manifests[name] = manifest
        return count

    def list_skill_manifests(self, category: str = None) -> List[dict]:
        """List file-based skills discovered from ``SKILL.md`` files."""
        result: List[dict] = []
        for manifest in self._manifests.values():
            if category and manifest.get("category") != category:
                continue
            result.append(dict(manifest))
        return sorted(result, key=lambda item: item.get("name", ""))

    def get_manifest(self, name: str) -> Optional[dict]:
        """Return a discovered ``SKILL.md`` manifest by name."""
        manifest = self._manifests.get(name)
        return dict(manifest) if manifest else None

    def validate_manifests(self) -> List[dict]:
        """Return validation results for currently discovered manifests."""
        return self.list_skill_manifests()

    def execute(self, agent, name: str, **kwargs) -> dict:
        """Execute a registered Python skill by name."""
        skill = self._skills.get(name)
        if skill is None:
            if name in self._manifests:
                return {
                    "success": False,
                    "result": None,
                    "message": f"Skill '{name}' is declarative and has no Python executor.",
                    "error_code": "SKILL_NOT_EXECUTABLE",
                }
            return {
                "success": False,
                "result": None,
                "message": f"Skill '{name}' is not registered.",
                "error_code": "SKILL_NOT_FOUND",
            }
        try:
            return skill.execute(agent, **kwargs)
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "message": f"Skill '{name}' execution failed: {exc}",
                "error_code": "SKILL_EXECUTION_FAILED",
            }

    def auto_discover(self) -> int:
        """扫描 skills/ 目录，导入所有模块并注册 SkillBase 子类。

        跳过以 '_' 开头的模块（如 __init__），对每个 .py 文件：
            1. 动态导入模块
            2. 查找模块中定义的 SkillBase 子类（排除基类本身）
            3. 实例化并注册

        Returns:
            新注册的技能数量
        """
        pkg_dir = Path(__file__).parent
        count = 0
        for py_file in sorted(pkg_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            mod_name = py_file.stem
            try:
                mod = importlib.import_module(
                    f".{mod_name}", package=__package__)
            except Exception as e:
                logging.warning(
                    f"自动发现: 导入技能模块 {mod_name} 失败: {e}",
                    exc_info=True,
                )
                continue
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name, None)
                if (obj is None or not inspect.isclass(obj)
                        or not issubclass(obj, SkillBase)
                        or obj is SkillBase
                        or obj.__module__ != mod.__name__):
                    continue
                try:
                    inst = obj()
                    if inst.name and inst.name not in self._skills:
                        self.register(inst)
                        count += 1
                except Exception as e:
                    logging.warning(
                        f"自动发现: 实例化技能 {attr_name} 失败: {e}",
                        exc_info=True,
                    )
                    continue
        return count

    def __len__(self) -> int:
        """已注册技能数量。"""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """检查技能是否已注册。"""
        return name in self._skills
