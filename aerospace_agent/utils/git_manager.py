"""Git 管理工具。

``GitManager`` 封装常用 git 操作，用 subprocess 调用 git 命令，捕获异常，
并在未配置时自动设置 ``user.email`` / ``user.name``，避免提交失败。
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

from aerospace_agent.local_runtime import run_command


class GitManager:
    """Git 操作封装。"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = str(repo_path)

    # ------------------------------------------------------------------
    # 内部：执行 git 子命令
    # ------------------------------------------------------------------
    def _run(self, args: List[str], check: bool = False,
             timeout: int = 60) -> Tuple[bool, str]:
        """执行 git 子命令，返回 (是否成功, 输出文本)。"""
        try:
            result = run_command(
                ["git"] + args,
                cwd=self.repo_path,
                timeout=timeout,
            )
            ok = result.ok
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

    # ------------------------------------------------------------------
    # 远程协作操作 (push / pull / fetch / remote)
    # ------------------------------------------------------------------
    def push(self, remote: str = "origin", branch: str = None,
             force: bool = False, tags: bool = False) -> str:
        """推送到远程仓库。

        Args:
            remote: 远程名称 (默认 origin)
            branch: 分支名 (为空则推送当前分支)
            force: 是否强制推送 (--force)
            tags: 是否推送标签 (--tags)

        Returns:
            操作结果描述
        """
        args = ["push"]
        if force:
            args.append("--force")
        if tags:
            args.append("--tags")
        args.append(remote)
        if branch:
            args.append(branch)
        ok, out = self._run(args, timeout=120)
        if ok:
            return f"推送成功: {remote}" + (f"/{branch}" if branch else "")
        return f"push 失败: {out}"

    def pull(self, remote: str = "origin", branch: str = None) -> str:
        """从远程仓库拉取并合并。

        Args:
            remote: 远程名称 (默认 origin)
            branch: 分支名 (为空则拉取当前分支)

        Returns:
            操作结果描述
        """
        args = ["pull", remote]
        if branch:
            args.append(branch)
        ok, out = self._run(args, timeout=120)
        if ok:
            return f"拉取成功: {remote}" + (f"/{branch}" if branch else "")
        return f"pull 失败: {out}"

    def fetch(self, remote: str = "origin", branch: str = None,
              all_remotes: bool = False) -> str:
        """从远程仓库获取 (不合并)。

        Args:
            remote: 远程名称 (默认 origin)
            branch: 分支名 (为空则获取所有分支)
            all_remotes: 是否获取所有远程 (--all)

        Returns:
            操作结果描述
        """
        args = ["fetch"]
        if all_remotes:
            args.append("--all")
        else:
            args.append(remote)
            if branch:
                args.append(branch)
        ok, out = self._run(args, timeout=120)
        if ok:
            return f"fetch 成功: {remote}" + (f"/{branch}" if branch else "")
        return f"fetch 失败: {out}"

    def add_remote(self, name: str, url: str) -> str:
        """添加远程仓库。

        Args:
            name: 远程名称 (如 origin)
            url: 远程仓库 URL

        Returns:
            操作结果描述
        """
        ok, out = self._run(["remote", "add", name, url])
        return f"远程 '{name}' 已添加" if ok else f"add_remote 失败: {out}"

    def list_remotes(self) -> str:
        """列出已配置的远程仓库。"""
        ok, out = self._run(["remote", "-v"])
        return out if ok else "无远程仓库配置"

    def clone(self, url: str, dest: str = None) -> str:
        """克隆远程仓库到指定目录。

        Args:
            url: 远程仓库 URL
            dest: 目标目录 (为空则使用仓库名)

        Returns:
            操作结果描述
        """
        args = ["clone", url]
        if dest:
            args.append(dest)
        ok, out = self._run(args, timeout=300)
        if ok:
            return f"克隆成功: {url}"
        return f"clone 失败: {out}"
