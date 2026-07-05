"""Astropy 接口工具 —— 时间、坐标与天文常数。

依赖库：astropy (Python 天文计算库)。

真实模式（astropy 可用）：
    - ``to_julian`` / ``from_julian`` 使用 ``astropy.time.Time``。
    - ``sidereal_time`` 使用 ``Time.sidereal_time``。
    - ``celestial_to_state`` 使用 ``astropy.coordinates.SkyCoord``。

回退模式（astropy 不可用）：
    - 自行实现儒略日转换（Meeus 算法）与 GMST 公式，注释给出推导。
    - 天球坐标 -> 笛卡尔状态向量用球坐标变换。
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseTool

# J2000 历元儒略日
JD_J2000 = 2451545.0
MJD_J2000 = 51544.5  # 约化儒略日


# ----------------------------------------------------------------------
# 回退模式：儒略日与恒星时公式（含推导注释）
# ----------------------------------------------------------------------
def _parse_date(date_str: str) -> tuple:
    """解析日期字符串 'YYYY-MM-DD [HH:MM:SS]' 返回 (year,month,day,h,m,s)。

    支持空格或 'T' 分隔日期与时间；时间部分可缺省（视为 0）。
    """
    s = date_str.strip().replace("T", " ")
    # 拆分日期与时间
    parts = s.split()
    date_part = parts[0]
    time_part = parts[1] if len(parts) > 1 else "00:00:00"

    y, m, d = [int(x) for x in date_part.split("-")]
    # 时间可能只有 HH 或 HH:MM 或 HH:MM:SS(.fff)
    tbits = time_part.split(":")
    hh = int(tbits[0]) if len(tbits) > 0 else 0
    mm = int(tbits[1]) if len(tbits) > 1 else 0
    ss = float(tbits[2]) if len(tbits) > 2 else 0.0
    return y, m, d, hh, mm, ss


def _to_julian_fallback(date_str: str) -> float:
    """公历日期 -> 儒略日（Meeus《天文算法》第7章算法）。

    推导要点：
        若月份 <= 2，视为上一年 13/14 月（保证历法连续）。
        引入格里高利修正项 B（1582-10-15 之后才需要）。
        JD = floor(365.25*(Y+4716)) + floor(30.6001*(M+1)) + D + B - 1524.5
    """
    y, m, d, hh, mm, ss = _parse_date(date_str)
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    # 格里高利历修正
    B = 2 - A + A // 4
    # 儒略日整数部分
    jd = (
        int(365.25 * (y + 4716))
        + int(30.6001 * (m + 1))
        + d
        + B
        - 1524.5
    )
    # 加上当日小数部分（UT 时分秒 -> 日）
    day_frac = (hh + mm / 60.0 + ss / 3600.0) / 24.0
    return jd + day_frac


def _from_julian_fallback(jd: float) -> str:
    """儒略日 -> 公历日期字符串（Meeus 逆算法）。

    推导要点：
        JD+0.5 分整数 Z 与小数 F；
        Z<2299161 用儒略历规则，否则引入格里高利修正 alpha；
        逆推 year/month/day 并从小数部分提取时分秒。
    """
    jd0 = jd + 0.5
    Z = int(jd0)
    F = jd0 - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - alpha // 4
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    day_int = int(day)
    frac = day - day_int
    total_sec = round(frac * 86400.0)
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return f"{year:04d}-{month:02d}-{day_int:02d} {hh:02d}:{mm:02d}:{ss:02d}"


def _gmst_hours(jd: float) -> float:
    """格林尼治平恒星时（小时），IAU 1982 单公式（Vallado）。

    推导：
        T = (JD - 2451545.0)/36525  (儒略世纪，含小数日)
        GMST(秒) = 67310.54841
                   + (876600*3600 + 8640184.812866) * T
                   + 0.093104 * T^2
                   - 6.2e-6 * T^3
        其中 876600*3600 项对应每世纪恒星日整数圈，
        67310.54841 s = 18.697374 h 即 J2000 历元的 GMST。
        该公式直接给出任意 UT 时刻的 GMST，无需分离 0h UT。
    """
    T = (jd - JD_J2000) / 36525.0
    theta_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    gmst = (theta_sec / 3600.0) % 24.0
    if gmst < 0:
        gmst += 24.0
    return gmst


def _sidereal_time_fallback(jd: float, longitude_deg: float) -> dict:
    """本地恒星时（小时与度），GMST + 经度。"""
    gmst_h = _gmst_hours(jd)
    # 本地恒星时 = GMST + 经度(东正)，单位小时
    lst_h = (gmst_h + longitude_deg / 15.0) % 24.0
    if lst_h < 0:
        lst_h += 24.0
    return {
        "gmst_hours": gmst_h,
        "gmst_degrees": gmst_h * 15.0 % 360.0,
        "lst_hours": lst_h,
        "lst_degrees": lst_h * 15.0,
        "longitude_deg": longitude_deg,
        "epoch_jd": jd,
    }


def _celestial_to_state_fallback(
    ra_deg: float, dec_deg: float, distance: float, epoch: float,
    pm_ra: float = 0.0, pm_dec: float = 0.0, radial_velocity: float = 0.0,
) -> dict:
    """天球坐标 (RA, Dec, 距离) -> 笛卡尔状态向量。

    推导（球坐标 -> 直角坐标）：
        x = d * cos(Dec) * cos(RA)
        y = d * cos(Dec) * sin(RA)
        z = d * sin(Dec)
    速度（若有自行 pm_ra/pm_dec(角秒/年) 与视向速度 RV(km/s)）：
        近似沿切向与径向分解，此处给出简化估计。
    """
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    cosd = math.cos(dec)
    pos = np.array([
        distance * cosd * math.cos(ra),
        distance * cosd * math.sin(ra),
        distance * math.sin(dec),
    ])
    # 速度：若提供自行则估算切向速度
    if (pm_ra or pm_dec or radial_velocity) and distance > 0:
        # 切向速度 = distance(km) * pm(arcsec/yr) / 206265 / (秒/年)
        sec_per_year = 365.25 * 86400.0
        # RA 方向切向（注意 cos(dec) 修正）
        v_ra = distance * (pm_ra / 206265.0) * cosd / sec_per_year
        v_dec = distance * (pm_dec / 206265.0) / sec_per_year
        # 切向单位向量
        e_ra = np.array([-math.sin(ra), math.cos(ra), 0.0])
        e_dec = np.array([-math.sin(dec) * math.cos(ra),
                          -math.sin(dec) * math.sin(ra), math.cos(dec)])
        e_r = pos / distance
        vel = e_ra * v_ra + e_dec * v_dec + e_r * radial_velocity
    else:
        vel = np.zeros(3)
    return {
        "position": pos.tolist(),
        "velocity": vel.tolist(),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "distance": distance,
        "epoch_jd": epoch,
        "frame": "ICRS",
    }


class AstropyTool(BaseTool):
    """Astropy 时间/坐标/常数工具。"""

    name = "astropy"
    description = "时间转换（儒略日）、恒星时、天球坐标转状态向量"
    library_name = "astropy"

    methods_schema = {
        "to_julian": {
            "params": {"date_str": "str"},
            "returns": "dict",
            "description": "公历日期字符串 -> 儒略日",
        },
        "from_julian": {
            "params": {"jd": "float"},
            "returns": "dict",
            "description": "儒略日 -> 公历日期字符串",
        },
        "sidereal_time": {
            "params": {"jd": "float", "longitude": "float"},
            "returns": "dict",
            "description": "计算格林尼治/本地恒星时",
        },
        "celestial_to_state": {
            "params": {"ra": "float", "dec": "float",
                       "distance": "float", "epoch": "float"},
            "returns": "dict",
            "description": "天球坐标 (RA,Dec,距离) -> 笛卡尔状态向量",
        },
    }

    # ------------------------------------------------------------------
    # 真实模式实现
    # ------------------------------------------------------------------
    def _to_julian_real(self, date_str: str) -> dict:
        from astropy.time import Time
        t = Time(date_str, format="iso", scale="utc")
        return {"jd": float(t.jd), "date_str": date_str, "mjd": float(t.mjd)}

    def _from_julian_real(self, jd: float) -> dict:
        from astropy.time import Time
        t = Time(jd, format="jd", scale="utc")
        return {"date_str": t.iso, "jd": jd, "mjd": float(t.mjd)}

    def _sidereal_time_real(self, jd: float, longitude_deg: float) -> dict:
        from astropy.time import Time
        import astropy.units as u
        t = Time(jd, format="jd", scale="utc")
        gmst = t.sidereal_time("mean", longitude=0.0 * u.deg)
        lst = t.sidereal_time("mean", longitude=longitude_deg * u.deg)
        return {
            "gmst_hours": float(gmst.hour),
            "gmst_degrees": float(gmst.degree),
            "lst_hours": float(lst.hour),
            "lst_degrees": float(lst.degree),
            "longitude_deg": longitude_deg,
            "epoch_jd": jd,
        }

    def _celestial_to_state_real(
        self, ra_deg: float, dec_deg: float, distance: float, epoch: float,
    ) -> dict:
        from astropy.coordinates import SkyCoord, CartesianRepresentation
        import astropy.units as u
        c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                     distance=distance * u.km, frame="icrs")
        cart = c.cartesian
        pos = [float(cart.x.value), float(cart.y.value), float(cart.z.value)]
        return {
            "position": pos,
            "velocity": [0.0, 0.0, 0.0],
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "distance": distance, "epoch_jd": epoch, "frame": "ICRS",
        }

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs) -> dict:
        if method == "to_julian":
            return self._call_to_julian(**kwargs)
        if method == "from_julian":
            return self._call_from_julian(**kwargs)
        if method == "sidereal_time":
            return self._call_sidereal_time(**kwargs)
        if method == "celestial_to_state":
            return self._call_celestial_to_state(**kwargs)
        return self._fail(f"未知方法: {method}", self.source,
                          f"可用方法: {self.list_methods()}")

    def _call_to_julian(self, date_str: str) -> dict:
        if self.is_available:
            try:
                return self._ok(self._to_julian_real(date_str), "real",
                                "astropy.time JD 转换完成。")
            except Exception as e:
                return self._fail(str(e), "real", "astropy JD 转换失败")
        try:
            jd = _to_julian_fallback(date_str)
            return self._ok(
                {"jd": jd, "date_str": date_str, "mjd": jd - 2400000.5},
                "fallback",
                "astropy 不可用，回退到 Meeus 儒略日公式。",
            )
        except Exception as e:
            return self._fail(str(e), "fallback", "回退 JD 转换失败")

    def _call_from_julian(self, jd: float) -> dict:
        if self.is_available:
            try:
                return self._ok(self._from_julian_real(jd), "real",
                                "astropy.time 逆转换完成。")
            except Exception as e:
                return self._fail(str(e), "real", "astropy 逆转换失败")
        try:
            date_str = _from_julian_fallback(jd)
            return self._ok(
                {"date_str": date_str, "jd": jd, "mjd": jd - 2400000.5},
                "fallback",
                "astropy 不可用，回退到 Meeus 逆算法。",
            )
        except Exception as e:
            return self._fail(str(e), "fallback", "回退逆转换失败")

    def _call_sidereal_time(self, jd: float, longitude: float) -> dict:
        if self.is_available:
            try:
                return self._ok(self._sidereal_time_real(jd, longitude), "real",
                                "astropy 恒星时计算完成。")
            except Exception as e:
                return self._fail(str(e), "real", "astropy 恒星时失败")
        try:
            res = _sidereal_time_fallback(jd, longitude)
            return self._ok(res, "fallback",
                            "astropy 不可用，回退到 GMST 公式（Meeus 第12章）。")
        except Exception as e:
            return self._fail(str(e), "fallback", "回退恒星时失败")

    def _call_celestial_to_state(
        self, ra: float, dec: float, distance: float, epoch: float,
    ) -> dict:
        if self.is_available:
            try:
                return self._ok(
                    self._celestial_to_state_real(ra, dec, distance, epoch),
                    "real", "astropy SkyCoord 坐标转换完成。",
                )
            except Exception as e:
                return self._fail(str(e), "real", "astropy 坐标转换失败")
        try:
            res = _celestial_to_state_fallback(ra, dec, distance, epoch)
            return self._ok(res, "fallback",
                            "astropy 不可用，回退到球坐标->直角坐标公式。")
        except Exception as e:
            return self._fail(str(e), "fallback", "回退坐标转换失败")


if __name__ == "__main__":
    tool = AstropyTool()
    print("工具信息:", {k: v for k, v in tool.get_info().items()
          if k != "methods_schema"})

    # to_julian: J2000 = '2000-01-01 12:00:00' -> 2451545.0
    print("\n--- to_julian ---")
    r = tool.call("to_julian", date_str="2000-01-01 12:00:00")
    print(r)
    assert abs(r["result"]["jd"] - 2451545.0) < 1e-6, "J2000 JD 校验失败"

    # from_julian
    print("\n--- from_julian ---")
    r2 = tool.call("from_julian", jd=2451545.0)
    print(r2)

    # 往返一致性
    r3 = tool.call("to_julian", date_str=r2["result"]["date_str"])
    print("往返 JD:", r3["result"]["jd"], "(应 = 2451545.0)")

    # sidereal_time: J2000 GMST 应约 18.697 小时
    print("\n--- sidereal_time ---")
    r4 = tool.call("sidereal_time", jd=2451545.0, longitude=116.4)
    print(r4["result"])
    print(f"GMST = {r4['result']['gmst_hours']:.5f} h (J2000 期望 ~18.6974 h)")

    # celestial_to_state
    print("\n--- celestial_to_state ---")
    r5 = tool.call("celestial_to_state", ra=90.0, dec=0.0,
                   distance=384400.0, epoch=2451545.0)
    print(r5["result"])
    # RA=90,Dec=0 -> pos 应在 +Y 方向
    assert abs(r5["result"]["position"][1] - 384400.0) < 1.0
    assert abs(r5["result"]["position"][0]) < 1e-6
    print(">>> 校验通过：RA=90°,Dec=0° -> +Y 方向")
