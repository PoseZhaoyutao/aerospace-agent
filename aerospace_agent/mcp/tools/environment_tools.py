"""环境检测工具 — 引擎可用性检查与参考 Demo 索引。

第一性原理（K2 环境感知）：
  1. MCP Server 启动时必须知道哪些引擎可用——不可用的绝不调用
  2. STK 不可用时返回 {available:False, reason:"..."} 而非崩溃
  3. Demo 索引只读第三方源码——绝不修改，绝不复制商业软件代码
  4. 所有结果 JSON 可序列化，供 LLM 决策
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..adapters import get_all_adapters
from ..safety import PathPolicy, check_license

#: 全部支持的引擎
_ALL_ENGINES = ["orekit", "gmat", "spiceypy", "astropy",
                "poliastro", "basilisk", "stk"]

#: Demo 来源及其默认扫描路径环境变量
_DEMO_SCAN_ENV = {
    "orekit": "OREKIT_DATA",
    "gmat": "GMAT_DATA",
    "spiceypy": "SPICE_KERNELS",
    "astropy": None,
    "poliastro": None,
    "basilisk": None,
    "stk": "STK_INSTALL_DIR",
}


def check_engine_availability(engines: Optional[List[str]] = None) -> Dict:
    """检查全部或指定引擎的可用性。

    Args:
        engines: 引擎名列表；None 表示检查全部 7 个
    Returns:
        {engine: {available, version, capabilities, data_path, license_status}}
    """
    targets = engines or _ALL_ENGINES
    adapters = get_all_adapters()
    result: Dict = {}

    for engine in targets:
        engine_lower = engine.lower().strip()
        adapter = adapters.get(engine_lower)
        if adapter is None:
            result[engine_lower] = {
                "available": False,
                "version": "unavailable",
                "capabilities": [],
                "data_path": "",
                "license_status": check_license(engine_lower),
                "reason": f"未知引擎: {engine}",
            }
            continue

        try:
            available = adapter.is_available()
            version = adapter.version() if available else "unavailable"
            caps = sorted(adapter.capabilities()) if available else []

            # 数据路径（从环境变量推断）
            env_var = _DEMO_SCAN_ENV.get(engine_lower)
            data_path = os.environ.get(env_var, "") if env_var else ""

            # 许可证状态
            license_info = check_license(engine_lower)

            if available:
                result[engine_lower] = {
                    "available": True,
                    "version": version,
                    "capabilities": caps,
                    "data_path": data_path,
                    "license_status": license_info,
                }
            else:
                reason = f"{engine_lower} 未安装或不可用"
                if engine_lower == "stk":
                    reason = (
                        "STK 未安装或许可证不可用。"
                        "STK 为商业软件，需安装 STK 并配置有效许可证。"
                    )
                result[engine_lower] = {
                    "available": False,
                    "version": "unavailable",
                    "capabilities": [],
                    "data_path": data_path,
                    "license_status": license_info,
                    "reason": reason,
                }
        except Exception as exc:
            result[engine_lower] = {
                "available": False,
                "version": "unavailable",
                "capabilities": [],
                "data_path": "",
                "license_status": check_license(engine_lower),
                "reason": f"检测异常: {exc}",
            }

    return result


def index_reference_demos(sources: Optional[List[str]] = None,
                          scan_paths: Optional[List[str]] = None) -> Dict:
    """扫描配置路径，索引各引擎的示例/测试/教程。

    只读操作——绝不修改第三方源码目录。
    对于商业软件（STK），仅构建元数据，绝不复制代码。

    Args:
        sources: 引擎来源列表；None 表示全部
        scan_paths: 自定义扫描路径列表；None 时从环境变量推断
    Returns:
        {total_indexed, demos: [{source, title, engine, task_type, path, metadata}]}
    """
    targets = sources or _ALL_ENGINES
    demos: List[Dict] = []

    for engine in targets:
        engine_lower = engine.lower().strip()
        paths = scan_paths or _resolve_scan_paths(engine_lower)
        for base_path in paths:
            if not base_path or not os.path.isdir(base_path):
                continue
            if not PathPolicy.is_allowed_read(base_path):
                continue
            demos.extend(_scan_directory(base_path, engine_lower))

    return {
        "total_indexed": len(demos),
        "demos": demos,
    }


def _resolve_scan_paths(engine: str) -> List[str]:
    """从环境变量解析引擎的默认扫描路径。"""
    env_var = _DEMO_SCAN_ENV.get(engine)
    if env_var:
        val = os.environ.get(env_var, "")
        return [val] if val else []
    return []


def _scan_directory(base_path: str, engine: str) -> List[Dict]:
    """递归扫描目录，提取工作流元数据（只读）。"""
    found: List[Dict] = []
    exts = {".py", ".m", ".script", ".gs", ".txt"}
    try:
        for root, _dirs, files in os.walk(base_path):
            for fname in files:
                if Path(fname).suffix.lower() not in exts:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    meta = _extract_metadata(fpath, engine)
                    if meta:
                        found.append(meta)
                except Exception:
                    continue
            # 限制扫描深度和数量，防止超大型目录卡死
            if len(found) > 500:
                break
    except Exception:
        pass
    return found


def _extract_metadata(fpath: str, engine: str) -> Optional[Dict]:
    """从文件内容提取工作流元数据（标题、任务类型等）。"""
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(8192)
    except Exception:
        return None

    title = Path(fpath).stem
    task_type = "unknown"

    # 从注释/文档字符串推断标题
    title_match = re.search(r'^[#""\']*\s*(?:Title|Demo|Example|名称)[:：]?\s*(.+)',
                            content, re.MULTILINE | re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip().strip('"\'.')

    # 推断任务类型
    lower = content.lower()
    if any(k in lower for k in ("propag", "orbit", "轨道", "传播")):
        task_type = "orbit_propagation"
    elif any(k in lower for k in ("access", "ground", "visibility", "可见")):
        task_type = "ground_access"
    elif any(k in lower for k in ("frame", "transform", "坐标")):
        task_type = "frame_transform"
    elif any(k in lower for k in ("ephemeris", "spice", "星历")):
        task_type = "ephemeris_query"
    elif any(k in lower for k in ("attitude", "姿态")):
        task_type = "attitude_dynamics"

    return {
        "source": engine,
        "title": title,
        "engine": engine,
        "task_type": task_type,
        "path": fpath,
        "metadata": {
            "file_size_bytes": os.path.getsize(fpath),
            "read_only": True,
            "commercial": engine == "stk",
        },
    }


__all__ = ["check_engine_availability", "index_reference_demos"]
