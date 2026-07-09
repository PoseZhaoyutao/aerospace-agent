"""许可检查 — 商业软件许可证验证。

第一性原理（K1 合规边界）：
  1. 开源引擎（orekit/spiceypy/astropy/poliastro/basilisk/gmat）视为已许可
  2. STK 等商业软件必须检查许可证——通过 COM 接口或环境变量
  3. 许可证不可用时返回结构化信息，绝不崩溃
  4. 所有检查结果可审计（licensed + details + method）
"""
from __future__ import annotations

import os
from typing import Dict

#: 开源引擎集合——无需许可证检查
_OPEN_SOURCE_ENGINES = {
    "orekit", "gmat", "spiceypy", "astropy", "poliastro", "basilisk",
}

#: 商业引擎集合——需要许可证
_COMMERCIAL_ENGINES = {"stk"}


def check_license(engine: str) -> Dict[str, object]:
    """检查指定引擎的许可证状态。

    Args:
        engine: 引擎名（orekit/gmat/spiceypy/astropy/poliastro/basilisk/stk）
    Returns:
        {engine, licensed, details, method} 字典
    """
    engine = engine.lower().strip()

    if engine in _OPEN_SOURCE_ENGINES:
        return {
            "engine": engine,
            "licensed": True,
            "details": "open_source — 该引擎为开源软件，无需商业许可证",
            "method": "static_allowlist",
        }

    if engine == "stk":
        return _check_stk_license()

    return {
        "engine": engine,
        "licensed": False,
        "details": f"未知引擎 '{engine}'，无法判断许可证状态",
        "method": "unknown_engine",
    }


def _check_stk_license() -> Dict[str, object]:
    """检查 STK 许可证——优先通过 COM 接口，其次检查环境变量。"""
    # 方法 1：检查 STK_LICENSE_FILE 环境变量
    license_env = os.environ.get("STK_LICENSE_FILE", "")
    if license_env and os.path.exists(license_env):
        return {
            "engine": "stk",
            "licensed": True,
            "details": f"检测到 STK 许可证文件: {license_env}",
            "method": "env_var_check",
        }

    # 方法 2：尝试通过 COM 接口检查（仅 Windows）
    try:
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()
        try:
            from win32com.client import Dispatch  # type: ignore
            app = Dispatch("STK.Application")
            app.Visible = False
            version = str(app.Version)
            app.Close()
            return {
                "engine": "stk",
                "licensed": True,
                "details": f"STK COM 接口可用，版本 {version}",
                "method": "com_interface",
            }
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        pass

    # 方法 3：检查 STK 安装目录
    stk_install = os.environ.get("STK_INSTALL_DIR", "")
    if stk_install and os.path.isdir(stk_install):
        return {
            "engine": "stk",
            "licensed": False,
            "details": (
                f"检测到 STK 安装目录 {stk_install}，但未找到有效许可证。"
                "请设置 STK_LICENSE_FILE 环境变量或确保 COM 许可服务可用。"
            ),
            "method": "install_dir_no_license",
        }

    return {
        "engine": "stk",
        "licensed": False,
        "details": "未检测到 STK 安装或许可证。STK 为商业软件，需要有效许可证。",
        "method": "not_found",
    }


def check_all_licenses() -> Dict[str, Dict[str, object]]:
    """检查全部引擎的许可证状态。"""
    all_engines = _OPEN_SOURCE_ENGINES | _COMMERCIAL_ENGINES
    return {e: check_license(e) for e in sorted(all_engines)}


__all__ = ["check_license", "check_all_licenses"]
