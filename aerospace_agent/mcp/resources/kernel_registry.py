"""KernelRegistry — SPICE 内核注册与路径验证。

第一性原理：SPICE 内核加载顺序和路径必须受控，
否则 LLM 可能加载恶意内核或产生错误的星历结果。
本注册表是内核路径白名单的唯一来源——任何 SPICE 加载操作
都必须先通过 ``validate_path`` 校验。

默认内核集覆盖闰秒、行星星历、地球参考系三类基础内核。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


#: 默认内核集——SPICE 通用基础内核
DEFAULT_KERNEL_SET: Dict[str, dict] = {
    "naif0012.tls": {
        "path": "naif0012.tls",
        "kernel_type": "lsk",  # leap seconds kernel 闰秒内核
        "targets": ["Earth", "Moon", "Sun", "SSB"],
        "description": "闰秒内核（UTC <-> TAI）",
    },
    "de440s.bsp": {
        "path": "de440s.bsp",
        "kernel_type": "spk",  # planetary ephemeris 行星星历
        "targets": [
            "Sun", "Mercury", "Venus", "Earth", "Mars",
            "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
            "Moon", "SSB",
        ],
        "description": "JPL DE440s 行星星历内核",
    },
    "earth_000101_240101_240724.bpc": {
        "path": "earth_000101_240101_240724.bpc",
        "kernel_type": "bpc",  # binary orientation 二进制定向
        "targets": ["Earth", "ITRF", "IAU_EARTH"],
        "description": "地球定向内核（ITRF <-> GCRF）",
    },
    "earth_000101_240101_240724.tf": {
        "path": "earth_000101_240101_240724.tf",
        "kernel_type": "fk",  # frame kernel 参考系内核
        "targets": ["Earth", "ITRF", "IAU_EARTH"],
        "description": "地球参考系内核",
    },
}


#: 内核类型加载优先级（lsk 必须最先加载）
_KERNEL_TYPE_ORDER: Dict[str, int] = {
    "lsk": 0, "spk": 1, "pck": 2, "fk": 3,
    "bpc": 4, "ik": 5, "ck": 6,
}


class KernelRegistry:
    """SPICE 内核注册表——管理内核文件、目标映射、路径验证。

    典型用法::

        reg = KernelRegistry()
        reg.load_from_config("kernels.json")
        paths = reg.get_kernels(targets=["Moon", "Earth"])
        assert reg.validate_path(paths[0])

    安全约束：``validate_path`` 是路径白名单闸门，
    所有 SPICE ``furnsh`` 调用前必须校验。
    """

    def __init__(self) -> None:
        self._kernels: Dict[str, dict] = {}
        self._authorized_paths: set[str] = set()
        # 载入默认内核集
        for name, info in DEFAULT_KERNEL_SET.items():
            self._kernels[name] = dict(info)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register_kernel(
        self,
        name: str,
        path: str,
        kernel_type: str = "spk",
        targets: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """注册一个内核文件。

        Args:
            name: 内核名（如 ``de440s.bsp``）
            path: 内核文件路径
            kernel_type: 内核类型（lsk/spk/fk/bpc/pck/ik/ck）
            targets: 该内核覆盖的天体/参考系列表
            description: 说明
        """
        self._kernels[name] = {
            "path": str(path),
            "kernel_type": kernel_type,
            "targets": list(targets or []),
            "description": description,
        }

    def load_from_config(self, config_path: Union[str, Path]) -> int:
        """从 JSON 配置文件批量加载内核注册。

        配置格式::

            {"kernels": [
                {"name": "de440s.bsp", "path": "...", "kernel_type": "spk",
                 "targets": ["Moon"], "description": "..."}
            ]}

        Returns:
            成功注册的内核数量
        """
        config_path = Path(config_path)
        if not config_path.exists():
            return 0
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 0
        count = 0
        for entry in config.get("kernels", []):
            name = entry.get("name", "")
            path = entry.get("path", "")
            if not name or not path:
                continue
            self.register_kernel(
                name=name,
                path=path,
                kernel_type=entry.get("kernel_type", "spk"),
                targets=entry.get("targets"),
                description=entry.get("description", ""),
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_kernels(self, targets: Optional[List[str]] = None) -> List[str]:
        """获取指定目标所需的内核文件路径列表。

        Args:
            targets: 目标天体/参考系列表；为 None 时返回所有内核路径

        Returns:
            内核文件路径列表（去重，按 lsk→spk→pck→fk→bpc 优先级排序）
        """
        if targets is None:
            return [info["path"] for info in self._kernels.values()]
        target_set = set(targets)
        matched: List[tuple] = []
        seen: set[str] = set()
        for info in self._kernels.values():
            ktargets = set(info.get("targets", []))
            if target_set & ktargets:
                path = info["path"]
                if path not in seen:
                    order = _KERNEL_TYPE_ORDER.get(info["kernel_type"], 9)
                    matched.append((order, path))
                    seen.add(path)
        matched.sort(key=lambda x: x[0])
        return [p for _, p in matched]

    def list_kernels(self) -> List[dict]:
        """列出所有已注册内核（含名称）。"""
        return [{"name": k, **v} for k, v in self._kernels.items()]

    # ------------------------------------------------------------------
    # 路径验证（安全闸门）
    # ------------------------------------------------------------------
    def validate_path(self, path: Union[str, Path]) -> bool:
        """验证路径是否在注册表或显式授权路径中。

        Args:
            path: 待验证的文件路径

        Returns:
            路径合法返回 True，否则 False

        这是 SPICE ``furnsh`` 前的必经校验，防止加载未授权内核。
        """
        try:
            target = str(Path(path).resolve())
        except (OSError, ValueError):
            target = str(path)
        # 检查注册表中的路径
        for info in self._kernels.values():
            try:
                if str(Path(info["path"]).resolve()) == target:
                    return True
            except (OSError, ValueError):
                if info["path"] == str(path):
                    return True
        # 检查显式授权路径
        for ap in self._authorized_paths:
            try:
                if str(Path(ap).resolve()) == target:
                    return True
            except (OSError, ValueError):
                if ap == str(path):
                    return True
        return False

    def authorize_path(self, path: Union[str, Path]) -> None:
        """显式授权一个路径（扩展白名单）。"""
        self._authorized_paths.add(str(path))

    def __len__(self) -> int:
        return len(self._kernels)

    def __contains__(self, name: object) -> bool:
        return name in self._kernels
