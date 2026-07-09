"""系统与环境工具集 (5 个)——环境变量 / 系统信息 / 计时 / KV 缓存。

设计原则：
  1. 全部使用标准库（os / sys / platform / time / json）
  2. ``set_env`` 仅作用于当前进程（os.environ），不修改系统级配置
  3. ``measure_time`` 在受限沙箱中执行代码并计时
  4. ``cache_get_set`` 提供进程内 KV 缓存，get/set 二合一
  5. 所有工具返回 dict / list / str（JSON 可序列化）

工具清单
--------
- get_env          : 获取环境变量
- set_env          : 设置环境变量（仅当前进程）
- get_system_info  : 获取系统信息（OS / Python 版本 / CPU / 内存）
- measure_time     : 测量代码执行时间
- cache_get_set    : 简单 KV 缓存（get / set 二合一）
"""
from __future__ import annotations

import json as _json
import os
import platform
import sys
import time
from typing import Any, Dict, Optional

from aerospace_agent.research_tools.base import register_tool


@register_tool(
    "get_env",
    "获取环境变量",
    "system_env",
    params=[
        {"name": "name", "type": "str", "description": "环境变量名"},
        {"name": "default", "type": "str", "description": "默认值",
         "required": False, "default": ""},
    ],
)
def get_env(name, default=""):
    value = os.environ.get(name, default)
    exists = name in os.environ
    return {
        "status": "success",
        "name": name,
        "value": value,
        "exists": exists,
    }


@register_tool(
    "set_env",
    "设置环境变量（仅当前进程，不修改系统级配置）",
    "system_env",
    params=[
        {"name": "name", "type": "str", "description": "环境变量名"},
        {"name": "value", "type": "str", "description": "环境变量值"},
    ],
)
def set_env(name, value):
    if not name:
        return {"status": "error", "reason": "环境变量名不能为空"}
    old_value = os.environ.get(name)
    os.environ[name] = str(value)
    return {
        "status": "success",
        "name": name,
        "value": str(value),
        "previous_value": old_value,
        "scope": "process",
    }


@register_tool(
    "get_system_info",
    "获取系统信息（OS / Python 版本 / CPU / 内存）",
    "system_env",
    params=[
        {"name": "include_memory", "type": "bool",
         "description": "是否尝试获取内存信息", "required": False,
         "default": True},
    ],
)
def get_system_info(include_memory=True):
    info: Dict[str, Any] = {
        "status": "success",
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    if include_memory:
        mem_info = _get_memory_info()
        info["memory"] = mem_info

    return info


def _get_memory_info() -> Dict[str, Any]:
    """跨平台尝试获取内存信息。"""
    mem: Dict[str, Any] = {"available": False}
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        mem = {
            "available": True,
            "source": "psutil",
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "used_bytes": vm.used,
            "percent": vm.percent,
        }
        return mem
    except ImportError:
        pass

    # Windows 回退
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            mem = {
                "available": True,
                "source": "GlobalMemoryStatusEx",
                "total_bytes": stat.ullTotalPhys,
                "available_bytes": stat.ullAvailPhys,
                "used_bytes": stat.ullTotalPhys - stat.ullAvailPhys,
                "percent": stat.dwMemoryLoad,
            }
        except Exception:
            pass
    else:
        # Linux 回退：读取 /proc/meminfo
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            data = {}
            for ln in lines:
                parts = ln.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    data[key] = int(val) * 1024  # kB → bytes
            mem = {
                "available": True,
                "source": "/proc/meminfo",
                "total_bytes": data.get("MemTotal"),
                "available_bytes": data.get("MemAvailable"),
                "used_bytes": (data.get("MemTotal", 0)
                               - data.get("MemAvailable", 0)),
            }
        except Exception:
            pass

    return mem


@register_tool(
    "measure_time",
    "测量 Python 代码执行时间（受限沙箱执行，返回耗时与输出）",
    "system_env",
    params=[
        {"name": "code", "type": "str", "description": "要计时的 Python 代码"},
        {"name": "repeat", "type": "int", "description": "重复执行次数",
         "required": False, "default": 1},
        {"name": "warmup", "type": "bool", "description": "是否预热一次",
         "required": False, "default": False},
    ],
)
def measure_time(code, repeat=1, warmup=False):
    if not isinstance(code, str) or not code.strip():
        return {"status": "error", "reason": "code 必须为非空字符串"}

    # 复用 code_execution 的安全沙箱
    try:
        from aerospace_agent.research_tools.code_execution import (
            _build_sandbox_globals, _to_jsonable,
        )
    except Exception:
        # 回退：极简受限命名空间
        import math
        import json
        import re as _re

        def _build_sandbox_globals():
            return {
                "__builtins__": {
                    "abs": abs, "all": all, "any": any, "bool": bool,
                    "dict": dict, "float": float, "int": int, "len": len,
                    "list": list, "max": max, "min": min, "print": print,
                    "range": range, "round": round, "set": set, "sorted": sorted,
                    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
                    "map": map, "filter": filter, "enumerate": enumerate,
                    "pow": pow, "repr": repr, "type": type, "isinstance": isinstance,
                },
                "math": math, "json": json, "re": _re,
            }

        def _to_jsonable(v):
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                return v
            return repr(v)

    import io
    from contextlib import redirect_stdout

    repeat = max(1, int(repeat))

    def _run_once():
        sandbox = _build_sandbox_globals()
        local_ns: Dict[str, Any] = {}
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(code, "<measure>", "exec"), sandbox, local_ns)
        return buf.getvalue(), local_ns

    try:
        if warmup:
            _run_once()

        times: List[float] = []
        last_stdout = ""
        last_vars: Dict[str, Any] = {}
        for _ in range(repeat):
            t0 = time.perf_counter()
            last_stdout, last_vars = _run_once()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        result_vars = {
            k: _to_jsonable(v)
            for k, v in last_vars.items()
            if not k.startswith("_")
        }
        return {
            "status": "success",
            "repeat": repeat,
            "times_seconds": [round(t, 6) for t in times],
            "total_seconds": round(sum(times), 6),
            "average_seconds": round(sum(times) / len(times), 6),
            "min_seconds": round(min(times), 6),
            "max_seconds": round(max(times), 6),
            "stdout": last_stdout,
            "variables": result_vars,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }


# 进程内 KV 缓存
_CACHE_STORE: Dict[str, Any] = {}


@register_tool(
    "cache_get_set",
    "简单 KV 缓存（get/set 二合一）：仅传 key 则 get，传 key+value 则 set",
    "system_env",
    params=[
        {"name": "key", "type": "str", "description": "缓存键"},
        {"name": "value", "type": "any",
         "description": "缓存值（传入则 set，省略则 get）",
         "required": False, "default": None},
        {"name": "ttl", "type": "int",
         "description": "存活秒数（仅 set 生效，<=0 表示永久）",
         "required": False, "default": 0},
        {"name": "action", "type": "str",
         "description": "显式操作：get/set/delete/clear/count",
         "required": False, "default": "auto"},
    ],
)
def cache_get_set(key, value=None, ttl=0, action="auto"):
    import time as _time

    action = (action or "auto").lower()

    # 清空全部缓存
    if action == "clear":
        n = len(_CACHE_STORE)
        _CACHE_STORE.clear()
        return {"status": "success", "action": "clear", "cleared": n}

    # 缓存条目计数
    if action == "count":
        return {"status": "success", "action": "count",
                "count": len(_CACHE_STORE)}

    # 删除指定 key
    if action == "delete":
        existed = key in _CACHE_STORE
        _CACHE_STORE.pop(key, None)
        return {"status": "success", "action": "delete", "key": key,
                "existed": existed}

    # 自动判定：传了 value 视为 set
    if action == "auto":
        action = "set" if value is not None else "get"

    if action == "set":
        _CACHE_STORE[key] = {
            "value": value,
            "expire_at": (_time.time() + ttl) if ttl and ttl > 0 else None,
        }
        return {
            "status": "success",
            "action": "set",
            "key": key,
            "ttl": ttl,
        }

    # action == "get"
    entry = _CACHE_STORE.get(key)
    if entry is None:
        return {
            "status": "success",
            "action": "get",
            "key": key,
            "hit": False,
            "value": None,
        }
    # 检查过期
    if entry.get("expire_at") is not None:
        if _time.time() > entry["expire_at"]:
            _CACHE_STORE.pop(key, None)
            return {
                "status": "success",
                "action": "get",
                "key": key,
                "hit": False,
                "value": None,
                "expired": True,
            }
    return {
        "status": "success",
        "action": "get",
        "key": key,
        "hit": True,
        "value": entry["value"],
    }


# List 已在 typing 中通过其它模块引用，此处确保可用
try:
    from typing import List
except ImportError:  # pragma: no cover
    pass


# ---- 模块自测 ----
if __name__ == "__main__":
    print("get_env:", get_env("PATH"))
    print("set_env:", set_env("MY_TEST_VAR", "hello"))
    print("get_env (after set):", get_env("MY_TEST_VAR"))
    print("get_system_info:", get_system_info(include_memory=False))
    print("measure_time:", measure_time("x = sum(range(1000))", repeat=3))
    print("cache_get_set (set):", cache_get_set("k1", value=42))
    print("cache_get_set (get):", cache_get_set("k1"))
    print("cache_get_set (count):", cache_get_set("", action="count"))
