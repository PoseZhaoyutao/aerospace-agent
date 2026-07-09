"""safety — 安全模块：许可检查、沙箱守卫、路径策略。

所有 MCP 工具的文件操作和引擎调用都必须经过 safety 层校验，
确保：
  1. 文件读写不越界（PathPolicy + SandboxGuard）
  2. 商业软件许可证合规（license_check）
  3. 不向第三方源码目录写入
"""
from __future__ import annotations

from .license_check import check_license, check_all_licenses
from .sandbox import SandboxGuard
from .path_policy import PathPolicy

__all__ = [
    "check_license",
    "check_all_licenses",
    "SandboxGuard",
    "PathPolicy",
]
