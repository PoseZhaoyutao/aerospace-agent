"""aerospace-agent 统一配置系统。

第一性原理：
  1. 所有配置集中管理，环境变量作为覆盖层
  2. 支持配置文件 (config.yaml) + 环境变量 + 默认值三层优先级
  3. 配置项按功能分组：LLM / Data / Engine / Safety / RAG / Observability
  4. 零外部依赖（不用 pydantic-settings，用 dataclass + os.environ）

优先级（从高到低）：
  1. 环境变量   —— 永远覆盖同名配置项
  2. 配置文件   —— ``Config.from_yaml("config.yaml")``
  3. 默认值     —— dataclass 字段默认值

环境变量约定（与现有 ``os.environ.get`` 调用保持一致，便于渐进迁移）：

    LLM
        AEROSPACE_LLM_API_KEY         云端 API 密钥
        AEROSPACE_LLM_BASE_URL        OpenAI 兼容 API 基址
        AEROSPACE_LLM_MODEL           模型名
        AEROSPACE_LOCAL_LLM_BASE_URL  本地推理服务基址
        AEROSPACE_LOCAL_LLM_MODEL     本地模型名
        AEROSPACE_LOCAL_LLM_API_KEY   本地服务 API 密钥
    Data
        AEROSPACE_DATA_DIR            数据根目录
    Engine
        GMAT_PATH                     GMAT 安装路径
        OREKIT_DATA                   Orekit 物理数据目录
        SPICE_KERNELS                 SPICE 内核目录
    Safety
        ASTRO_DYNAMICS_WORKSPACE      工作空间根（路径白名单）
        STK_LICENSE_FILE              STK 许可文件
        STK_INSTALL_DIR               STK 安装目录
    RAG
        CSTCLOUD_API_KEY              中国科技云 API 密钥
        CSTCLOUD_BASE_URL             中国科技云基址
    Observability
        AEROSPACE_LOG_LEVEL           日志级别
        AEROSPACE_LOG_JSON            JSON 结构化日志开关 (1/0)

典型用法::

    from aerospace_agent.config import get_config
    cfg = get_config()
    print(cfg.llm.model)

或显式从配置文件加载（环境变量仍会覆盖文件值）::

    from aerospace_agent.config import Config
    cfg = Config.from_yaml("config.yaml")

约束：
  - 零外部依赖（仅标准库 + PyYAML；PyYAML 缺失时 ``from_env``/``get_config`` 仍可用）
  - 环境变量优先级高于配置文件
  - 不修改现有代码（现有代码继续用 ``os.environ.get``，新代码可用 ``get_config()``）
"""
from __future__ import annotations

import dataclasses
import logging
import os
import typing
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)

# PyYAML 是唯一允许的第三方依赖；缺失时仅禁用 YAML 相关能力。
try:  # pragma: no cover - 取决于运行环境
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


__all__ = [
    "LLMConfig",
    "DataConfig",
    "EngineConfig",
    "SafetyConfig",
    "RAGConfig",
    "ObservabilityConfig",
    "Config",
    "get_config",
    "reset_config",
]


# =============================================================================
# 子配置 dataclass
# =============================================================================


@dataclass
class LLMConfig:
    """大语言模型配置。"""

    api_key: str = ""  # AEROSPACE_LLM_API_KEY
    base_url: str = ""  # AEROSPACE_LLM_BASE_URL
    model: str = ""  # AEROSPACE_LLM_MODEL
    local_base_url: str = "http://127.0.0.1:8000/v1"  # AEROSPACE_LOCAL_LLM_BASE_URL
    local_model: str = "qwen3-vl"  # AEROSPACE_LOCAL_LLM_MODEL
    local_api_key: str = ""  # AEROSPACE_LOCAL_LLM_API_KEY


@dataclass
class DataConfig:
    """数据与输出目录配置。"""

    data_dir: str = "data"  # AEROSPACE_DATA_DIR
    reports_dir: str = "reports"
    loop_runs_dir: str = "data/loop_runs"


@dataclass
class EngineConfig:
    """轨道计算引擎配置。"""

    gmat_path: str = ""  # GMAT_PATH
    orekit_data: str = ""  # OREKIT_DATA
    spice_kernels: str = ""  # SPICE_KERNELS


@dataclass
class SafetyConfig:
    """安全与许可证配置。"""

    workspace: str = ""  # ASTRO_DYNAMICS_WORKSPACE
    stk_license: str = ""  # STK_LICENSE_FILE
    stk_install_dir: str = ""  # STK_INSTALL_DIR


@dataclass
class RAGConfig:
    """检索增强（RAG）相关配置。"""

    cstcloud_api_key: str = ""  # CSTCLOUD_API_KEY
    cstcloud_base_url: str = ""  # CSTCLOUD_BASE_URL


@dataclass
class ObservabilityConfig:
    """可观测性（日志）配置。"""

    log_level: str = "WARNING"  # AEROSPACE_LOG_LEVEL
    log_json: bool = False  # AEROSPACE_LOG_JSON


# =============================================================================
# 字段 <-> 环境变量映射表
# =============================================================================

# 点分字段路径 -> 环境变量名。
# 未列出的字段（如 data.reports_dir）没有专属环境变量，
# 只能通过配置文件或默认值提供。
_ENV_MAP: Dict[str, str] = {
    "llm.api_key": "AEROSPACE_LLM_API_KEY",
    "llm.base_url": "AEROSPACE_LLM_BASE_URL",
    "llm.model": "AEROSPACE_LLM_MODEL",
    "llm.local_base_url": "AEROSPACE_LOCAL_LLM_BASE_URL",
    "llm.local_model": "AEROSPACE_LOCAL_LLM_MODEL",
    "llm.local_api_key": "AEROSPACE_LOCAL_LLM_API_KEY",
    "data.data_dir": "AEROSPACE_DATA_DIR",
    "engine.gmat_path": "GMAT_PATH",
    "engine.orekit_data": "OREKIT_DATA",
    "engine.spice_kernels": "SPICE_KERNELS",
    "safety.workspace": "ASTRO_DYNAMICS_WORKSPACE",
    "safety.stk_license": "STK_LICENSE_FILE",
    "safety.stk_install_dir": "STK_INSTALL_DIR",
    "rag.cstcloud_api_key": "CSTCLOUD_API_KEY",
    "rag.cstcloud_base_url": "CSTCLOUD_BASE_URL",
    "observability.log_level": "AEROSPACE_LOG_LEVEL",
    "observability.log_json": "AEROSPACE_LOG_JSON",
}

# 分组名 -> 子配置 dataclass 类型（用于类型推断与字段遍历）。
_SUB_CONFIG_TYPES: Dict[str, type] = {
    "llm": LLMConfig,
    "data": DataConfig,
    "engine": EngineConfig,
    "safety": SafetyConfig,
    "rag": RAGConfig,
    "observability": ObservabilityConfig,
}


# =============================================================================
# 内部工具函数
# =============================================================================


def _resolve_type(dotted_path: str) -> Any:
    """返回点分字段路径声明的类型（如 ``llm.log_json`` -> ``bool``）。

    无法解析时回退到 ``str``，保证宽松兼容。
    """
    group, name = dotted_path.split(".", 1)
    sub_cls = _SUB_CONFIG_TYPES.get(group)
    if sub_cls is None:
        return str
    try:
        hints = typing.get_type_hints(sub_cls)
    except Exception:  # pragma: no cover - 防御性
        return str
    return hints.get(name, str)


def _coerce(value: str, target_type: Any) -> Any:
    """将环境变量字符串按目标类型做轻量转换。"""
    if target_type is bool:
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if target_type is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def _set_attr(config: "Config", dotted_path: str, value: Any) -> None:
    """按点分路径写入子配置字段（如 ``llm.api_key``）。"""
    group, name = dotted_path.split(".", 1)
    setattr(getattr(config, group), name, value)


def _get_attr(config: "Config", dotted_path: str) -> Any:
    """按点分路径读取子配置字段。"""
    group, name = dotted_path.split(".", 1)
    return getattr(getattr(config, group), name)


def _apply_env_overrides(config: "Config") -> "Config":
    """将环境变量覆盖到已有 Config 实例上（就地修改并返回）。

    仅当环境变量已设置且非空时才覆盖，与现有
    ``os.environ.get("VAR") or default`` 行为一致。
    """
    for dotted, env_var in _ENV_MAP.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        _set_attr(config, dotted, _coerce(raw, _resolve_type(dotted)))
    return config


# =============================================================================
# 顶层 Config
# =============================================================================


@dataclass
class Config:
    """aerospace-agent 顶层配置，聚合各功能分组子配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置（未设置的项保留 dataclass 默认值）。"""
        return _apply_env_overrides(cls())

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置，环境变量仍可覆盖文件值。

        三层优先级：默认值 < 配置文件 < 环境变量。
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML config support. "
                "Install it with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # 第二层：配置文件覆盖默认值。
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML 配置文件顶层应为映射(dict)，实际为 {type(data).__name__}: {path}"
            )
        for group, sub_cls in _SUB_CONFIG_TYPES.items():
            section = data.get(group)
            if not isinstance(section, dict):
                if section is not None:
                    _logger.warning(
                        "YAML 配置分组 '%s' 应为 dict,实际为 %s,已跳过",
                        group, type(section).__name__,
                    )
                continue
            sub_cfg = getattr(config, group)
            for field_name in sub_cls.__dataclass_fields__:
                if field_name in section:
                    field_type = _resolve_type(f"{group}.{field_name}")
                    setattr(sub_cfg, field_name,
                            _coerce(section[field_name], field_type))

        # 第三层：环境变量覆盖文件值。
        return _apply_env_overrides(config)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def to_env_dict(self) -> Dict[str, str]:
        """转为环境变量字典（env_var_name -> 字符串值）。

        布尔值转为 ``"1"``/``"0``；仅包含 ``_ENV_MAP`` 中映射的字段。
        """
        result: Dict[str, str] = {}
        for dotted, env_var in _ENV_MAP.items():
            value = _get_attr(self, dotted)
            if isinstance(value, bool):
                result[env_var] = "1" if value else "0"
            elif value is None:
                result[env_var] = ""
            else:
                result[env_var] = str(value)
        return result

    def to_yaml(self, path: str) -> None:
        """将当前配置写入 YAML 文件。"""
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML config support. "
                "Install it with: pip install pyyaml"
            )
        data = {
            group: asdict(getattr(self, group))
            for group in _SUB_CONFIG_TYPES
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )


# =============================================================================
# 全局单例
# =============================================================================

_CONFIG_SINGLETON: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例。

    首次调用时从环境变量加载（``Config.from_env()``）并缓存；
    后续调用直接返回缓存实例。测试中可用 :func:`reset_config` 清除缓存。
    """
    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is None:
        _CONFIG_SINGLETON = Config.from_env()
    return _CONFIG_SINGLETON


def reset_config() -> None:
    """清除全局配置单例缓存（主要供测试使用）。"""
    global _CONFIG_SINGLETON
    _CONFIG_SINGLETON = None
