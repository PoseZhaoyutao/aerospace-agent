"""Git 管理工具。

``GitManager`` 封装常用 git 操作，用 subprocess 调用 git 命令，捕获异常，
并在未配置时自动设置 ``user.email`` / ``user.name``，避免提交失败。
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple


class GitManager:
    """Git 操作封装。"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = str(repo_path)

    # ------------------------------------------------------------------
    # 内部：执行 git 子命令
    # ------------------------------------------------------------------
    def _run(self, args: List[str], check: bool = False) -> Tuple[bool, str]:
        """执行 git 子命令，返回 (是否成功, 输出文本)。"""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            ok = result.returncode == 0
            out = result.stdout if ok else (result.stderr or result.stdout)
            return ok, out.strip()
        except FileNotFoundError:
            return False, "未找到 git 可执行文件，请确认已安装 git"
        except subprocess.TimeoutExpired:
            return False, "git 命令执行超时"
        except Exception as e:
            return False, f"git 执行异常: {e}"

    def _ensure_identity(self) -> None:
        """确保 git user.email / user.name 已配置，否则自动设置默认值。"""
        ok_email, email = self._run(["config", "user.email"])
        if not ok_email or not email:
            self._run(["config", "user.email", "aerospace-agent@local"])
        ok_name, name = self._run(["config", "user.name"])
        if not ok_name or not name:
            self._run(["config", "user.name", "Aerospace Agent"])

    # ------------------------------------------------------------------
    # 公开操作
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 git 是否可用。"""
        ok, _ = self._run(["--version"])
        return ok

    def init(self, repo_path: str = None) -> str:
        """初始化 git 仓库。

        Args:
            repo_path: 仓库路径（为空则使用当前 repo_path）

        Returns:
            操作结果描述
        """
        if repo_path:
            self.repo_path = str(repo_path)
            Path(self.repo_path).mkdir(parents=True, exist_ok=True)
        ok, out = self._run(["init"])
        self._ensure_identity()
        return out if ok else f"init 失败: {out}"

    def add(self, pathspec: str = ".") -> str:
        """将文件加入暂存区。"""
        ok, out = self._run(["add", pathspec])
        return out if ok else f"add 失败: {out}"

    def commit(self, msg: str) -> str:
        """提交暂存区变更。"""
        self._ensure_identity()
        ok, out = self._run(["commit", "-m", msg])
        if not ok:
            # 可能没有变更可提交
            return f"commit: {out}"
        return out

    def status(self) -> str:
        """查看工作区状态（简短格式）。"""
        ok, out = self._run(["status", "-s"])
        return out if ok else f"status 失败: {out}"

    def log(self, n: int = 10) -> str:
        """查看最近 n 条提交日志。"""
        ok, out = self._run(["log", f"-{n}", "--oneline"])
        return out if ok else f"log 失败: {out}"

    def create_branch(self, name: str) -> str:
        """创建新分支。"""
        ok, out = self._run(["branch", name])
        return f"分支 '{name}' 已创建" if ok else f"create_branch 失败: {out}"

    def checkout(self, name: str) -> str:
        """切换分支。"""
        ok, out = self._run(["checkout", name])
        return f"已切换到 '{name}'" if ok else f"checkout 失败: {out}"
