"""技能与工作流自进化引擎。

第一性原理 (K5): 版本化技能注册表 + 成功模式自动沉淀。
execute() 返回 metrics，达标自动升级为模板；
失败记录回退基准。

进化策略:
    1. 技能版本化: 每个技能有版本号和性能指标
    2. 成功模式沉淀: 连续 N 次成功 → 自动标记为 "proven" 模板
    3. 失败回退: 连续 M 次失败 → 回退到上一个稳定版本
    4. 工作流模板: 成功的工具调用序列自动保存为 workflow 模板
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


# 进化数据存储路径
DEFAULT_EVOLUTION_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "evolution.json"
)


@dataclass
class SkillMetric:
    """技能执行指标。"""
    skill_name: str
    version: str = "1.0.0"
    total_runs: int = 0
    success_runs: int = 0
    failure_runs: int = 0
    avg_duration_ms: float = 0.0
    last_run_at: str = ""
    last_error: str = ""
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    status: str = "active"  # active | proven | deprecated | failed
    template: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowTemplate:
    """工作流模板 — 成功的工具调用序列。"""
    name: str
    description: str
    intent: str
    tool_sequence: List[Dict[str, Any]]  # [{tool_name, args, expected_result_pattern}]
    success_count: int = 0
    total_uses: int = 0
    created_at: str = ""
    updated_at: str = ""
    version: str = "1.0.0"


class EvolutionEngine:
    """技能与工作流自进化引擎。

    管理技能版本、性能指标、自动升级/回退、工作流模板沉淀。

    Attributes:
        db_path: 进化数据持久化路径
        skills: 技能指标字典 {skill_name: SkillMetric}
        workflows: 工作流模板字典 {name: WorkflowTemplate}
        auto_upgrade_threshold: 连续成功 N 次自动升级
        auto_rollback_threshold: 连续失败 M 次自动回退
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        auto_upgrade_threshold: int = 5,
        auto_rollback_threshold: int = 3,
    ):
        """初始化进化引擎。

        Args:
            db_path: 持久化路径
            auto_upgrade_threshold: 自动升级阈值（连续成功次数）
            auto_rollback_threshold: 自动回退阈值（连续失败次数）
        """
        self.db_path = db_path or DEFAULT_EVOLUTION_DB
        self.auto_upgrade_threshold = auto_upgrade_threshold
        self.auto_rollback_threshold = auto_rollback_threshold
        self.skills: Dict[str, SkillMetric] = {}
        self.workflows: Dict[str, WorkflowTemplate] = {}
        self._load()

    def _load(self) -> None:
        """从持久化存储加载进化数据。"""
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for name, sdata in data.get("skills", {}).items():
                self.skills[name] = SkillMetric(**sdata)

            for name, wdata in data.get("workflows", {}).items():
                self.workflows[name] = WorkflowTemplate(**wdata)
        except Exception:
            pass

    def _save(self) -> None:
        """持久化进化数据。"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "skills": {n: s.__dict__ for n, s in self.skills.items()},
            "workflows": {n: w.__dict__ for n, w in self.workflows.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- 技能进化 ----

    def register_skill(
        self,
        name: str,
        version: str = "1.0.0",
    ) -> SkillMetric:
        """注册新技能（如不存在则创建）。

        Args:
            name: 技能名称
            version: 初始版本

        Returns:
            SkillMetric 实例
        """
        if name not in self.skills:
            self.skills[name] = SkillMetric(
                skill_name=name,
                version=version,
            )
            self._save()
        return self.skills[name]

    def record_execution(
        self,
        skill_name: str,
        success: bool,
        duration_ms: float,
        error: str = "",
        result_summary: Optional[Dict[str, Any]] = None,
    ) -> SkillMetric:
        """记录一次技能执行。

        Args:
            skill_name: 技能名称
            success: 是否成功
            duration_ms: 执行耗时（毫秒）
            error: 错误信息（失败时）
            result_summary: 结果摘要（成功时，用于模板沉淀）

        Returns:
            更新后的 SkillMetric
        """
        if skill_name not in self.skills:
            self.register_skill(skill_name)

        metric = self.skills[skill_name]
        metric.total_runs += 1
        metric.last_run_at = datetime.now().isoformat()

        if success:
            metric.success_runs += 1
            metric.consecutive_successes += 1
            metric.consecutive_failures = 0
            metric.last_error = ""

            # 自动升级检查
            if (
                metric.status == "active"
                and metric.consecutive_successes >= self.auto_upgrade_threshold
            ):
                metric.status = "proven"
                if result_summary:
                    metric.template = result_summary

            # 更新平均耗时
            if metric.avg_duration_ms == 0:
                metric.avg_duration_ms = duration_ms
            else:
                metric.avg_duration_ms = (
                    metric.avg_duration_ms * 0.9 + duration_ms * 0.1
                )
        else:
            metric.failure_runs += 1
            metric.consecutive_failures += 1
            metric.consecutive_successes = 0
            metric.last_error = error

            # 自动回退检查
            if (
                metric.status in ("active", "proven")
                and metric.consecutive_failures >= self.auto_rollback_threshold
            ):
                metric.status = "deprecated"

        self._save()
        return metric

    def get_skill_status(self, skill_name: str) -> Optional[SkillMetric]:
        """获取技能状态。

        Args:
            skill_name: 技能名称

        Returns:
            SkillMetric 或 None
        """
        return self.skills.get(skill_name)

    def get_all_skills(self) -> Dict[str, SkillMetric]:
        """获取所有技能指标。

        Returns:
            {skill_name: SkillMetric}
        """
        return dict(self.skills)

    def get_proven_skills(self) -> Dict[str, SkillMetric]:
        """获取已验证的稳定技能。

        Returns:
            {skill_name: SkillMetric}
        """
        return {
            n: s for n, s in self.skills.items()
            if s.status == "proven"
        }

    # ---- 工作流进化 ----

    def register_workflow(
        self,
        name: str,
        description: str,
        intent: str,
        tool_sequence: List[Dict[str, Any]],
    ) -> WorkflowTemplate:
        """注册或更新工作流模板。

        Args:
            name: 工作流名称
            description: 描述
            intent: 关联意图
            tool_sequence: 工具调用序列

        Returns:
            WorkflowTemplate
        """
        now = datetime.now().isoformat()
        if name in self.workflows:
            wf = self.workflows[name]
            wf.tool_sequence = tool_sequence
            wf.updated_at = now
            wf.total_uses += 1
        else:
            wf = WorkflowTemplate(
                name=name,
                description=description,
                intent=intent,
                tool_sequence=tool_sequence,
                created_at=now,
                updated_at=now,
            )
            self.workflows[name] = wf

        self._save()
        return wf

    def record_workflow_success(self, name: str) -> None:
        """记录工作流成功执行。

        Args:
            name: 工作流名称
        """
        if name in self.workflows:
            wf = self.workflows[name]
            wf.success_count += 1
            wf.total_uses += 1
            wf.updated_at = datetime.now().isoformat()
            self._save()

    def get_workflows_for_intent(self, intent: str) -> List[WorkflowTemplate]:
        """获取指定意图的已验证工作流模板。

        Args:
            intent: 意图类型

        Returns:
            工作流模板列表（按成功率排序）
        """
        matching = [
            w for w in self.workflows.values()
            if w.intent == intent and w.success_count > 0
        ]
        matching.sort(
            key=lambda w: w.success_count / max(w.total_uses, 1),
            reverse=True,
        )
        return matching

    def get_all_workflows(self) -> Dict[str, WorkflowTemplate]:
        """获取所有工作流模板。

        Returns:
            {name: WorkflowTemplate}
        """
        return dict(self.workflows)

    # ---- 进化统计 ----

    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化引擎摘要统计。

        Returns:
            统计字典
        """
        total_runs = sum(s.total_runs for s in self.skills.values())
        total_success = sum(s.success_runs for s in self.skills.values())
        proven_count = len(self.get_proven_skills())
        deprecated_count = sum(
            1 for s in self.skills.values() if s.status == "deprecated"
        )

        return {
            "total_skills": len(self.skills),
            "total_runs": total_runs,
            "overall_success_rate": total_success / max(total_runs, 1),
            "proven_skills": proven_count,
            "deprecated_skills": deprecated_count,
            "workflow_templates": len(self.workflows),
            "auto_upgrade_threshold": self.auto_upgrade_threshold,
            "auto_rollback_threshold": self.auto_rollback_threshold,
        }


def create_evolution_engine(
    db_path: Optional[str] = None,
) -> EvolutionEngine:
    """创建进化引擎的工厂函数。

    Args:
        db_path: 持久化路径

    Returns:
        EvolutionEngine 实例
    """
    return EvolutionEngine(db_path=db_path)


# New reversible transaction API.  Keep the historical ``EvolutionEngine``
# above intact for callers that only record skill metrics.
from .services.evolution import EvolutionService
from .services.evolution_policy import EvolutionPolicy, Eligibility, parse_llm_proposal
from .services.evolution_validators import ValidationResult

__all__ = [
    "SkillMetric", "WorkflowTemplate", "EvolutionEngine", "create_evolution_engine",
    "EvolutionService", "EvolutionPolicy", "Eligibility", "ValidationResult", "parse_llm_proposal",
]
