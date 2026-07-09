"""路径策略 — 控制文件读写边界，防止越权访问。

第一性原理（K1 安全边界）：
  1. 读操作仅允许白名单根目录（引擎安装路径、kernel 目录、工作区）
  2. 写操作仅允许工作区内——绝不向第三方源码目录写入
  3. kernel 路径必须经 kernel_registry 验证，杜绝任意文件加载
  4. 所有路径先规范化（resolve symlinks）再比较，防止路径穿越攻击
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


class PathPolicy:
    """路径访问策略守卫。

    所有 MCP 工具的文件操作必须先经过 PathPolicy 校验。
    """

    #: 允许读取的根目录（引擎数据、kernel、用户工作区等）
    # K5-H9: 移除 ~ (用户主目录)，防止读取 ~/.ssh/id_rsa 等敏感文件
    ALLOWED_READ_ROOTS: List[str] = [
        os.environ.get("GMAT_DATA", ""),
        os.environ.get("SPICE_KERNELS", ""),
        os.environ.get("OREKIT_DATA", ""),
        os.environ.get("ASTRO_DYNAMICS_WORKSPACE", ""),
        os.path.join(os.getcwd(), "data"),       # 项目数据目录
        os.path.join(os.getcwd(), "reports"),     # 报告目录
    ]

    #: 工作区根（写操作的唯一合法区域）
    WORKSPACE_ROOT: str = os.environ.get(
        "ASTRO_DYNAMICS_WORKSPACE",
        os.path.expanduser("~/.astro_dynamics_workspace"),
    )

    @classmethod
    def _normalize(cls, path: str) -> str:
        """规范化路径（解析符号链接、转为绝对路径）。"""
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(Path(path).absolute())

    @classmethod
    def is_allowed_read(cls, path: str) -> bool:
        """判断路径是否在允许读取的白名单根目录内。"""
        norm = cls._normalize(path)
        for root in cls.ALLOWED_READ_ROOTS:
            if not root:
                continue
            root_norm = cls._normalize(root)
            try:
                Path(norm).relative_to(root_norm)
                return True
            except ValueError:
                continue
        return False

    @classmethod
    def is_allowed_write(cls, path: str, workspace: str) -> bool:
        """判断路径是否在工作区内（写操作唯一合法区域）。"""
        norm = cls._normalize(path)
        ws_norm = cls._normalize(workspace)
        try:
            Path(norm).relative_to(ws_norm)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_kernel_path(cls, path: str,
                             kernel_registry: Optional[dict] = None) -> bool:
        """验证 kernel 文件路径是否在注册表中或合法 kernel 目录内。

        Args:
            path: kernel 文件路径
            kernel_registry: 已注册 kernel 字典 {name: path}，可选
        """
        norm = cls._normalize(path)
        # 若提供了注册表，优先匹配
        if kernel_registry:
            for reg_path in kernel_registry.values():
                if cls._normalize(str(reg_path)) == norm:
                    return True
        # 否则检查是否在 SPICE_KERNELS / OREKIT_DATA 等合法目录内
        for env_var in ("SPICE_KERNELS", "OREKIT_DATA"):
            root = os.environ.get(env_var, "")
            if root:
                try:
                    Path(norm).relative_to(cls._normalize(root))
                    return True
                except ValueError:
                    pass
        return cls.is_allowed_read(path)


__all__ = ["PathPolicy"]
