"""GMAT 脚本执行工具 — 在沙箱工作区内运行 GMAT 脚本。

第一性原理（K1 安全执行）：
  1. GMAT 脚本只能在指定工作区内运行——绝不执行任意路径的脚本
  2. 输出文件收集限定在工作区内
  3. GMAT 不可用时返回 {status:"unavailable"}，不崩溃
  4. 所有文件操作经 SandboxGuard 校验
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from aerospace_agent.local_runtime import run_command

from ..adapters import get_adapter
from ..safety import SandboxGuard, PathPolicy


def run_gmat_script(script_text: Optional[str] = None,
                    script_path: Optional[str] = None,
                    workspace: Optional[str] = None) -> Dict:
    """在工作区内运行 GMAT 脚本。

    Args:
        script_text: GMAT 脚本文本（直接提供时写入工作区临时文件执行）
        script_path: 已有 GMAT 脚本路径（复制到工作区后执行）
        workspace: 工作区目录（None 时使用默认工作区）
    Returns:
        {stdout, stderr, output_files, parsed_report, engine, workspace}
    """
    guard = SandboxGuard(workspace)
    ws = guard.ensure_workspace()

    # 检查 GMAT 可用性
    adapter = get_adapter("gmat")
    if not adapter.is_available():
        return {
            "status": "unavailable",
            "reason": (
                "GMAT 未安装或不可用。请安装 GMAT 并设置 GMAT_PATH 环境变量。"
            ),
            "engine": "gmat",
            "workspace": ws,
        }

    # 确定 GMAT 可执行文件（与 GMATAdapter 一致，使用 GMAT_PATH 环境变量）
    gmat_bin = _find_gmat_binary()
    if not gmat_bin:
        return {
            "status": "unavailable",
            "reason": (
                "未找到 GMAT 可执行文件。"
                "请设置 GMAT_PATH 环境变量指向 GMAT 可执行文件或其安装目录。"
            ),
            "engine": "gmat",
            "workspace": ws,
        }

    # 准备脚本文件
    if script_text:
        script_file = guard.safe_write_path("gmat_run.script", ws)
        try:
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script_text)
        except Exception as exc:
            return _error(f"脚本写入失败: {exc}", ws)
    elif script_path:
        try:
            script_file = guard.prepare_workspace_copy(script_path, ws)
        except Exception as exc:
            return _error(f"脚本复制失败: {exc}", ws)
    else:
        return _error("必须提供 script_text 或 script_path", ws)

    # 执行 GMAT
    try:
        result = run_command(
            [gmat_bin, "-r", script_file, "-x"],
            cwd=ws,
            timeout=300,
        )
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        return _error("GMAT 执行超时（300s）", ws)
    except Exception as exc:
        return _error(f"GMAT 执行失败: {exc}", ws)

    # 收集输出文件
    output_files = _collect_output_files(ws, script_file)

    # 解析报告
    parsed_report = _parse_gmat_report(stdout, output_files)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "output_files": output_files,
        "parsed_report": parsed_report,
        "engine": "gmat",
        "engine_version": adapter.version(),
        "workspace": ws,
        "script_file": script_file,
        "units": "SI (m, m/s, s, deg)",
        "status": "success" if return_code == 0 else "failed",
    }


def _find_executable(name: str) -> bool:
    """检查可执行文件是否存在（PATH 或绝对路径）。"""
    from shutil import which
    if os.path.isabs(name):
        return os.path.isfile(name)
    return which(name) is not None


def _find_gmat_binary() -> Optional[str]:
    """定位 GMAT 可执行文件——与 GMATAdapter._find_gmat_binary 逻辑一致。

    优先 GMAT_PATH 环境变量（可指向可执行文件或 bin 目录）。
    """
    env_path = os.environ.get("GMAT_PATH")
    if not env_path:
        return None
    p = Path(env_path)
    candidates = [p] if p.is_file() else []
    if p.is_dir():
        for n in ("gmat", "gmat.exe", "GmatConsole"):
            candidates.append(p / "bin" / n)
            candidates.append(p / n)
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _collect_output_files(workspace: str,
                          script_file: str) -> List[Dict]:
    """收集工作区内的输出文件。"""
    outputs = []
    ws_path = Path(workspace)
    for f in ws_path.iterdir():
        if f.is_file() and str(f.resolve()) != str(Path(script_file).resolve()):
            if f.suffix.lower() in (".report", ".txt", ".csv", ".orb", ".eph"):
                outputs.append({
                    "filename": f.name,
                    "path": str(f.resolve()),
                    "size_bytes": f.stat().st_size,
                })
    return outputs


def _parse_gmat_report(stdout: str, output_files: List[Dict]) -> Dict:
    """解析 GMAT 输出报告（简化版）。"""
    report: Dict = {"summary": "", "data_tables": [], "errors": []}

    # 从 stdout 提取关键信息
    lines = stdout.strip().split("\n") if stdout else []
    error_lines = [ln for ln in lines if "error" in ln.lower()
                   or "exception" in ln.lower()]
    report["errors"] = error_lines[:50]

    # 尝试读取 .report 文件
    for of in output_files:
        if of["filename"].endswith(".report"):
            try:
                with open(of["path"], "r", encoding="utf-8",
                          errors="ignore") as f:
                    content = f.read(65536)
                report["data_tables"].append({
                    "source": of["filename"],
                    "lines": content.count("\n"),
                    "preview": content[:500],
                })
            except Exception:
                pass

    if not error_lines:
        report["summary"] = "GMAT 执行完成，未检测到错误"
    else:
        report["summary"] = f"GMAT 执行完成，检测到 {len(error_lines)} 条错误/警告"

    return report


def _error(reason: str, workspace: str) -> Dict:
    return {
        "status": "error",
        "reason": reason,
        "engine": "gmat",
        "workspace": workspace,
        "stdout": "",
        "stderr": "",
        "output_files": [],
        "parsed_report": {},
    }


__all__ = ["run_gmat_script"]
