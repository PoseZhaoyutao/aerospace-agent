"""沙箱守卫 — 确保所有文件操作限定在工作区内。

第一性原理（K1 安全边界）：
  1. 所有写操作（脚本生成、输出文件）必须在工作区内
  2. 第三方源码只读——绝不向 GMAT/STK 安装目录写入
  3. 需要操作第三方文件时，先复制到工作区再操作（prepare_workspace_copy）
  4. validate_path 是所有文件 I/O 的前置闸门
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .path_policy import PathPolicy


class SandboxGuard:
    """工作区沙箱守卫。

    确保 MCP 工具的所有文件操作不会逃逸出工作区边界，
    也不会修改任何第三方软件安装目录。
    """

    def __init__(self, workspace: str | None = None):
        self.workspace = workspace or PathPolicy.WORKSPACE_ROOT

    def validate_path(self, path: str, workspace: str | None = None) -> bool:
        """验证路径是否在指定工作区内。

        Args:
            path: 待验证路径
            workspace: 工作区根目录（默认用 self.workspace）
        Returns:
            True 表示安全（在工作区内）
        """
        ws = workspace or self.workspace
        return PathPolicy.is_allowed_write(path, ws)

    def ensure_workspace(self, workspace: str | None = None) -> str:
        """确保工作区目录存在，不存在则创建。

        Returns:
            工作区绝对路径
        """
        ws = workspace or self.workspace
        Path(ws).mkdir(parents=True, exist_ok=True)
        return str(Path(ws).resolve())

    def prepare_workspace_copy(self, source_path: str,
                               workspace: str | None = None) -> str:
        """将源文件复制到工作区内，返回工作区副本路径。

        用于需要修改/运行第三方脚本的场景——绝不修改原始文件。

        Args:
            source_path: 原始文件路径（只读）
            workspace: 目标工作区
        Returns:
            工作区副本的绝对路径
        Raises:
            FileNotFoundError: 源文件不存在
            PermissionError: 源路径不在允许读取范围内
        """
        ws = workspace or self.workspace
        ws_dir = Path(self.ensure_workspace(ws))
        src = Path(source_path)

        if not src.exists():
            raise FileNotFoundError(f"源文件不存在: {source_path}")
        if not PathPolicy.is_allowed_read(source_path):
            raise PermissionError(f"源路径不在允许读取范围内: {source_path}")

        dest = ws_dir / src.name
        shutil.copy2(str(src), str(dest))
        return str(dest.resolve())

    def safe_write_path(self, filename: str,
                        workspace: str | None = None) -> str:
        """生成工作区内安全写入路径（用于输出文件）。

        Args:
            filename: 文件名（不含目录）
            workspace: 工作区根目录
        Returns:
            工作区内完整路径
        """
        ws = workspace or self.workspace
        ws_dir = Path(self.ensure_workspace(ws))
        return str((ws_dir / filename).resolve())


__all__ = ["SandboxGuard"]
