"""aerospace_agent.skills — 技能模块（可复用能力单元）。

技能（Skill）是 Agent 可调用的封装化能力单元，每个技能实现
SkillBase 接口，通过 SkillRegistry 统一注册、检索与执行。

主要导出
--------
* :class:`SkillBase`                技能抽象基类
* :class:`SkillRegistry`            技能注册表（注册/检索/执行/自动发现）
* :class:`ContextManagementSkill`   上下文管理技能（CEO 三层压缩/卸载）
* :class:`MemoryRecallSkill`        记忆召回技能（长期记忆检索）
* :class:`KnowledgeRetrievalSkill`  知识检索技能（多源路由 RAG）
* :class:`LoopOrchestrationSkill`   Loop 编排技能（八阶段自主交付循环）
* :class:`ReportGenerationSkill`    报告生成技能（自包含 HTML 报告）
"""
from __future__ import annotations

from .base import SkillBase
from .manifest import SkillManifest, discover_skill_manifests, validate_skill_manifest
from .registry import SkillRegistry
from .context_skill import ContextManagementSkill
from .memory_skill import MemoryRecallSkill
from .rag_skill import KnowledgeRetrievalSkill
from .loop_skill import LoopOrchestrationSkill
from .report_skill import ReportGenerationSkill

__all__ = [
    "SkillBase",
    "SkillManifest",
    "discover_skill_manifests",
    "validate_skill_manifest",
    "SkillRegistry",
    "ContextManagementSkill",
    "MemoryRecallSkill",
    "KnowledgeRetrievalSkill",
    "LoopOrchestrationSkill",
    "ReportGenerationSkill",
]

__version__ = "0.1.0"
