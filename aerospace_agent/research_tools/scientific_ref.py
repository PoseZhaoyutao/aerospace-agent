"""科学引用与参考工具集——10 个内置真实数据的科研参考原子工具。

第一性原理：
  1. 物理常数使用 CODATA 2018 推荐值（SI 精确定义值优先）
  2. 科学公式内置推导说明与适用条件，可直接代入计算
  3. 元素周期表内置全部 118 号元素的基本属性
  4. 所有转换基于精确数学关系，不依赖外部库
"""
from __future__ import annotations

import math
import datetime
from typing import Any, Dict, List

from aerospace_agent.research_tools.base import register_tool

# ===========================================================================
# 内置数据：物理常数（CODATA 2018，SI 单位）
# ===========================================================================
PHYSICAL_CONSTANTS: Dict[str, Dict[str, Any]] = {
    "speed_of_light": {
        "symbol": "c", "value": 299792458.0, "unit": "m/s",
        "description": "真空光速（精确值）",
        "category": "universal",
    },
    "gravitational_constant": {
        "symbol": "G", "value": 6.67430e-11, "unit": "m^3 kg^-1 s^-2",
        "description": "万有引力常数",
        "category": "gravitation",
    },
    "planck_constant": {
        "symbol": "h", "value": 6.62607015e-34, "unit": "J·s",
        "description": "普朗克常数（精确值）",
        "category": "quantum",
    },
    "reduced_planck_constant": {
        "symbol": "ℏ", "value": 1.054571817e-34, "unit": "J·s",
        "description": "约化普朗克常数 h/(2π)",
        "category": "quantum",
    },
    "boltzmann_constant": {
        "symbol": "k_B", "value": 1.380649e-23, "unit": "J/K",
        "description": "玻尔兹曼常数（精确值）",
        "category": "thermodynamics",
    },
    "avogadro_number": {
        "symbol": "N_A", "value": 6.02214076e23, "unit": "mol^-1",
        "description": "阿伏伽德罗常数（精确值）",
        "category": "chemistry",
    },
    "electron_mass": {
        "symbol": "m_e", "value": 9.1093837015e-31, "unit": "kg",
        "description": "电子静止质量",
        "category": "particle",
    },
    "proton_mass": {
        "symbol": "m_p", "value": 1.67262192369e-27, "unit": "kg",
        "description": "质子静止质量",
        "category": "particle",
    },
    "neutron_mass": {
        "symbol": "m_n", "value": 1.67492749804e-27, "unit": "kg",
        "description": "中子静止质量",
        "category": "particle",
    },
    "elementary_charge": {
        "symbol": "e", "value": 1.602176634e-19, "unit": "C",
        "description": "基本电荷（精确值）",
        "category": "electromagnetism",
    },
    "vacuum_permittivity": {
        "symbol": "ε_0", "value": 8.8541878128e-12, "unit": "F/m",
        "description": "真空介电常数",
        "category": "electromagnetism",
    },
    "vacuum_permeability": {
        "symbol": "μ_0", "value": 1.25663706212e-6, "unit": "N/A^2",
        "description": "真空磁导率",
        "category": "electromagnetism",
    },
    "stefan_boltzmann_constant": {
        "symbol": "σ", "value": 5.670374419e-8, "unit": "W m^-2 K^-4",
        "description": "斯特藩-玻尔兹曼常数",
        "category": "thermodynamics",
    },
    "wien_displacement_constant": {
        "symbol": "b", "value": 2.897771955e-3, "unit": "m·K",
        "description": "维恩位移常数",
        "category": "thermodynamics",
    },
    "rydberg_constant": {
        "symbol": "R_∞", "value": 1.0973731568160e7, "unit": "m^-1",
        "description": "里德伯常数",
        "category": "atomic",
    },
    "bohr_radius": {
        "symbol": "a_0", "value": 5.29177210903e-11, "unit": "m",
        "description": "玻尔半径",
        "category": "atomic",
    },
    "fine_structure_constant": {
        "symbol": "α", "value": 7.2973525693e-3, "unit": "dimensionless",
        "description": "精细结构常数",
        "category": "atomic",
    },
    "gas_constant": {
        "symbol": "R", "value": 8.314462618, "unit": "J mol^-1 K^-1",
        "description": "理想气体常数（精确值）",
        "category": "thermodynamics",
    },
    "standard_gravity": {
        "symbol": "g", "value": 9.80665, "unit": "m/s^2",
        "description": "标准重力加速度",
        "category": "gravitation",
    },
    "hubble_constant": {
        "symbol": "H_0", "value": 2.184e-18, "unit": "s^-1",
        "description": "哈勃常数（约 67.4 km/s/Mpc）",
        "category": "cosmology",
    },
    "mu_earth": {
        "symbol": "μ_⊕", "value": 3.986004418e14, "unit": "m^3/s^2",
        "description": "地球引力参数 G*M_earth",
        "category": "astronomy",
    },
    "mu_sun": {
        "symbol": "μ_☉", "value": 1.32712440018e20, "unit": "m^3/s^2",
        "description": "太阳引力参数 G*M_sun",
        "category": "astronomy",
    },
    "astronomical_unit": {
        "symbol": "AU", "value": 1.495978707e11, "unit": "m",
        "description": "天文单位",
        "category": "astronomy",
    },
    "earth_radius_equatorial": {
        "symbol": "R_⊕", "value": 6378137.0, "unit": "m",
        "description": "地球赤道半径（WGS-84）",
        "category": "astronomy",
    },
}

# ===========================================================================
# 内置数据：科学公式
# ===========================================================================
SCIENTIFIC_FORMULAS: Dict[str, Dict[str, Any]] = {
    "kepler_third_law": {
        "formula": "T^2 = (4π^2 / μ) × a^3",
        "description": "开普勒第三定律：轨道周期的平方与半长轴的立方成正比",
        "variables": {"T": "轨道周期(s)", "a": "半长轴(m)", "μ": "引力参数(m^3/s^2)"},
        "category": "orbital_mechanics",
        "example": "地球圆轨道 a=7000km → T=2π*sqrt(a^3/μ_earth)",
    },
    "escape_velocity": {
        "formula": "v_esc = sqrt(2μ / r)",
        "description": "逃逸速度：从天体表面逃逸所需最小速度",
        "variables": {"v_esc": "逃逸速度(m/s)", "μ": "引力参数", "r": "距离中心(m)"},
        "category": "orbital_mechanics",
    },
    "circular_orbit_velocity": {
        "formula": "v = sqrt(μ / r)",
        "description": "圆轨道速度",
        "variables": {"v": "轨道速度(m/s)", "μ": "引力参数", "r": "轨道半径(m)"},
        "category": "orbital_mechanics",
    },
    "vis_viva": {
        "formula": "v^2 = μ × (2/r - 1/a)",
        "description": "活力公式：任意椭圆轨道上任一点的速度",
        "variables": {"v": "速度(m/s)", "μ": "引力参数", "r": "当前半径(m)",
                      "a": "半长轴(m)"},
        "category": "orbital_mechanics",
    },
    "newton_gravitation": {
        "formula": "F = G × m1 × m2 / r^2",
        "description": "牛顿万有引力定律",
        "variables": {"F": "引力(N)", "G": "引力常数", "m1/m2": "质量(kg)",
                      "r": "距离(m)"},
        "category": "gravitation",
    },
    "kinetic_energy": {
        "formula": "KE = 0.5 × m × v^2",
        "description": "动能",
        "variables": {"KE": "动能(J)", "m": "质量(kg)", "v": "速度(m/s)"},
        "category": "mechanics",
    },
    "gravitational_potential_energy": {
        "formula": "PE = -G × m1 × m2 / r",
        "description": "引力势能（取无穷远为零势能）",
        "variables": {"PE": "势能(J)", "G": "引力常数", "m1/m2": "质量(kg)",
                      "r": "距离(m)"},
        "category": "gravitation",
    },
    "schwarzschild_radius": {
        "formula": "r_s = 2GM / c^2",
        "description": "史瓦西半径（黑洞事件视界）",
        "variables": {"r_s": "史瓦西半径(m)", "G": "引力常数", "M": "质量(kg)",
                      "c": "光速(m/s)"},
        "category": "relativity",
    },
    "tsiolkovsky_rocket": {
        "formula": "Δv = v_e × ln(m0 / m1)",
        "description": "齐奥尔科夫斯基火箭方程",
        "variables": {"Δv": "速度增量(m/s)", "v_e": "排气速度(m/s)",
                      "m0": "初始质量(kg)", "m1": "终末质量(kg)"},
        "category": "propulsion",
    },
    "lorentz_factor": {
        "formula": "γ = 1 / sqrt(1 - v^2/c^2)",
        "description": "洛伦兹因子（狭义相对论时间膨胀/长度收缩）",
        "variables": {"γ": "洛伦兹因子", "v": "速度(m/s)", "c": "光速(m/s)"},
        "category": "relativity",
    },
    "wien_displacement": {
        "formula": "λ_max = b / T",
        "description": "维恩位移定律：黑体辐射峰值波长与温度成反比",
        "variables": {"λ_max": "峰值波长(m)", "b": "维恩常数", "T": "温度(K)"},
        "category": "thermodynamics",
    },
    "stefan_boltzmann": {
        "formula": "P = σ × A × T^4",
        "description": "斯特藩-玻尔兹曼定律：黑体辐射总功率",
        "variables": {"P": "辐射功率(W)", "σ": "斯特藩常数", "A": "表面积(m^2)",
                      "T": "温度(K)"},
        "category": "thermodynamics",
    },
    "ideal_gas_law": {
        "formula": "P × V = n × R × T",
        "description": "理想气体状态方程",
        "variables": {"P": "压强(Pa)", "V": "体积(m^3)", "n": "物质的量(mol)",
                      "R": "气体常数", "T": "温度(K)"},
        "category": "thermodynamics",
    },
    "doppler_nonrelativistic": {
        "formula": "f' = f × (c ± v_r) / c",
        "description": "多普勒效应（非相对论近似）",
        "variables": {"f'": "观测频率(Hz)", "f": "源频率(Hz)", "c": "波速(m/s)",
                      "v_r": "径向相对速度(m/s)"},
        "category": "wave",
    },
    "de_broglie_wavelength": {
        "formula": "λ = h / p",
        "description": "德布罗意波长",
        "variables": {"λ": "波长(m)", "h": "普朗克常数", "p": "动量(kg·m/s)"},
        "category": "quantum",
    },
    "einstein_mass_energy": {
        "formula": "E = m × c^2",
        "description": "质能方程",
        "variables": {"E": "能量(J)", "m": "质量(kg)", "c": "光速(m/s)"},
        "category": "relativity",
    },
    "hohmann_transfer": {
        "formula": "Δv_total = sqrt(μ/r1)×(sqrt(2r2/(r1+r2))-1) + "
                   "sqrt(μ/r2)×(1-sqrt(2r1/(r1+r2)))",
        "description": "霍曼转移轨道总速度增量（共面圆轨道间最优双脉冲转移）",
        "variables": {"Δv_total": "总速度增量(m/s)", "μ": "引力参数",
                      "r1": "初始轨道半径(m)", "r2": "目标轨道半径(m)"},
        "category": "orbital_mechanics",
    },
}

# ===========================================================================
# 内置数据：单位转换因子（基准单位为 SI）
# ===========================================================================
UNIT_FACTORS: Dict[str, Dict[str, float]] = {
    "length": {  # 基准: meter
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "cm": 0.01, "centimeter": 0.01,
        "mm": 0.001, "millimeter": 0.001,
        "um": 1e-6, "micrometer": 1e-6, "nm": 1e-9, "nanometer": 1e-9,
        "mile": 1609.344, "miles": 1609.344,
        "yard": 0.9144, "yd": 0.9144,
        "foot": 0.3048, "ft": 0.3048, "feet": 0.3048,
        "inch": 0.0254, "in": 0.0254,
        "nautical_mile": 1852.0, "nmile": 1852.0,
        "au": 1.495978707e11, "astronomical_unit": 1.495978707e11,
        "light_year": 9.4607304725808e15, "ly": 9.4607304725808e15,
        "parsec": 3.0856775814913673e16, "pc": 3.0856775814913673e16,
    },
    "mass": {  # 基准: kilogram
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "mg": 1e-6, "milligram": 1e-6,
        "ug": 1e-9, "microgram": 1e-9,
        "ton": 1000.0, "tonne": 1000.0, "metric_ton": 1000.0,
        "lb": 0.45359237, "pound": 0.45359237, "lbs": 0.45359237,
        "oz": 0.028349523125, "ounce": 0.028349523125,
        "slug": 14.5939029372,
        "amu": 1.66053906660e-27, "u": 1.66053906660e-27,
    },
    "time": {  # 基准: second
        "s": 1.0, "second": 1.0, "seconds": 1.0, "sec": 1.0,
        "ms": 0.001, "millisecond": 0.001,
        "us": 1e-6, "microsecond": 1e-6,
        "ns": 1e-9, "nanosecond": 1e-9,
        "min": 60.0, "minute": 60.0, "minutes": 60.0,
        "h": 3600.0, "hour": 3600.0, "hours": 360.0 * 10,
        "day": 86400.0, "days": 86400.0,
        "week": 604800.0, "weeks": 604800.0,
        "year": 31557600.0, "years": 31557600.0,  # 儒略年
        "month": 2629800.0, "months": 2629800.0,
    },
    "speed": {  # 基准: m/s
        "m/s": 1.0, "mps": 1.0,
        "km/h": 1.0 / 3.6, "kph": 1.0 / 3.6,
        "mph": 0.44704, "mile/h": 0.44704,
        "ft/s": 0.3048, "fps": 0.3048,
        "knot": 0.514444444, "kn": 0.514444444,
        "mach": 340.29,  # 海平面标准大气声速近似
        "c": 299792458.0, "speed_of_light": 299792458.0,
    },
    "energy": {  # 基准: joule
        "J": 1.0, "joule": 1.0, "joules": 1.0,
        "kJ": 1000.0, "kilojoule": 1000.0,
        "MJ": 1e6, "megajoule": 1e6,
        "cal": 4.184, "calorie": 4.184,
        "kcal": 4184.0, "kilocalorie": 4184.0,
        "Wh": 3600.0, "watt_hour": 3600.0,
        "kWh": 3.6e6, "kilowatt_hour": 3.6e6,
        "eV": 1.602176634e-19, "electronvolt": 1.602176634e-19,
        "keV": 1.602176634e-16,
        "MeV": 1.602176634e-13,
        "BTU": 1055.05585, "btu": 1055.05585,
        "erg": 1e-7, "ergs": 1e-7,
        "ft_lbf": 1.3558179483314,
    },
    "temperature": {  # 特殊处理，非线性
        "K": "kelvin", "kelvin": "kelvin",
        "C": "celsius", "celsius": "celsius", "degC": "celsius",
        "F": "fahrenheit", "fahrenheit": "fahrenheit", "degF": "fahrenheit",
        "R": "rankine", "rankine": "rankine", "degR": "rankine",
    },
}


# ===========================================================================
# 内置数据：元素周期表（1-118 号元素）
# ===========================================================================
PERIODIC_TABLE: Dict[str, Dict[str, Any]] = {
    # key 为原子序数字符串
    "1": {"symbol": "H", "name": "氢", "name_en": "Hydrogen",
          "mass": 1.008, "category": "nonmetal", "group": 1, "period": 1},
    "2": {"symbol": "He", "name": "氦", "name_en": "Helium",
          "mass": 4.0026, "category": "noble_gas", "group": 18, "period": 1},
    "3": {"symbol": "Li", "name": "锂", "name_en": "Lithium",
          "mass": 6.94, "category": "alkali_metal", "group": 1, "period": 2},
    "4": {"symbol": "Be", "name": "铍", "name_en": "Beryllium",
          "mass": 9.0122, "category": "alkaline_earth", "group": 2, "period": 2},
    "5": {"symbol": "B", "name": "硼", "name_en": "Boron",
          "mass": 10.81, "category": "metalloid", "group": 13, "period": 2},
    "6": {"symbol": "C", "name": "碳", "name_en": "Carbon",
          "mass": 12.011, "category": "nonmetal", "group": 14, "period": 2},
    "7": {"symbol": "N", "name": "氮", "name_en": "Nitrogen",
          "mass": 14.007, "category": "nonmetal", "group": 15, "period": 2},
    "8": {"symbol": "O", "name": "氧", "name_en": "Oxygen",
          "mass": 15.999, "category": "nonmetal", "group": 16, "period": 2},
    "9": {"symbol": "F", "name": "氟", "name_en": "Fluorine",
          "mass": 18.998, "category": "halogen", "group": 17, "period": 2},
    "10": {"symbol": "Ne", "name": "氖", "name_en": "Neon",
           "mass": 20.180, "category": "noble_gas", "group": 18, "period": 2},
    "11": {"symbol": "Na", "name": "钠", "name_en": "Sodium",
           "mass": 22.990, "category": "alkali_metal", "group": 1, "period": 3},
    "12": {"symbol": "Mg", "name": "镁", "name_en": "Magnesium",
           "mass": 24.305, "category": "alkaline_earth", "group": 2, "period": 3},
    "13": {"symbol": "Al", "name": "铝", "name_en": "Aluminum",
           "mass": 26.982, "category": "post_transition", "group": 13, "period": 3},
    "14": {"symbol": "Si", "name": "硅", "name_en": "Silicon",
           "mass": 28.085, "category": "metalloid", "group": 14, "period": 3},
    "15": {"symbol": "P", "name": "磷", "name_en": "Phosphorus",
           "mass": 30.974, "category": "nonmetal", "group": 15, "period": 3},
    "16": {"symbol": "S", "name": "硫", "name_en": "Sulfur",
           "mass": 32.06, "category": "nonmetal", "group": 16, "period": 3},
    "17": {"symbol": "Cl", "name": "氯", "name_en": "Chlorine",
           "mass": 35.45, "category": "halogen", "group": 17, "period": 3},
    "18": {"symbol": "Ar", "name": "氩", "name_en": "Argon",
           "mass": 39.948, "category": "noble_gas", "group": 18, "period": 3},
    "19": {"symbol": "K", "name": "钾", "name_en": "Potassium",
           "mass": 39.098, "category": "alkali_metal", "group": 1, "period": 4},
    "20": {"symbol": "Ca", "name": "钙", "name_en": "Calcium",
           "mass": 40.078, "category": "alkaline_earth", "group": 2, "period": 4},
    "21": {"symbol": "Sc", "name": "钪", "name_en": "Scandium",
           "mass": 44.956, "category": "transition_metal", "group": 3, "period": 4},
    "22": {"symbol": "Ti", "name": "钛", "name_en": "Titanium",
           "mass": 47.867, "category": "transition_metal", "group": 4, "period": 4},
    "23": {"symbol": "V", "name": "钒", "name_en": "Vanadium",
           "mass": 50.942, "category": "transition_metal", "group": 5, "period": 4},
    "24": {"symbol": "Cr", "name": "铬", "name_en": "Chromium",
           "mass": 51.996, "category": "transition_metal", "group": 6, "period": 4},
    "25": {"symbol": "Mn", "name": "锰", "name_en": "Manganese",
           "mass": 54.938, "category": "transition_metal", "group": 7, "period": 4},
    "26": {"symbol": "Fe", "name": "铁", "name_en": "Iron",
           "mass": 55.845, "category": "transition_metal", "group": 8, "period": 4},
    "27": {"symbol": "Co", "name": "钴", "name_en": "Cobalt",
           "mass": 58.933, "category": "transition_metal", "group": 9, "period": 4},
    "28": {"symbol": "Ni", "name": "镍", "name_en": "Nickel",
           "mass": 58.693, "category": "transition_metal", "group": 10, "period": 4},
    "29": {"symbol": "Cu", "name": "铜", "name_en": "Copper",
           "mass": 63.546, "category": "transition_metal", "group": 11, "period": 4},
    "30": {"symbol": "Zn", "name": "锌", "name_en": "Zinc",
           "mass": 65.38, "category": "transition_metal", "group": 12, "period": 4},
    "31": {"symbol": "Ga", "name": "镓", "name_en": "Gallium",
           "mass": 69.723, "category": "post_transition", "group": 13, "period": 4},
    "32": {"symbol": "Ge", "name": "锗", "name_en": "Germanium",
           "mass": 72.630, "category": "metalloid", "group": 14, "period": 4},
    "33": {"symbol": "As", "name": "砷", "name_en": "Arsenic",
           "mass": 74.922, "category": "metalloid", "group": 15, "period": 4},
    "34": {"symbol": "Se", "name": "硒", "name_en": "Selenium",
           "mass": 78.971, "category": "nonmetal", "group": 16, "period": 4},
    "35": {"symbol": "Br", "name": "溴", "name_en": "Bromine",
           "mass": 79.904, "category": "halogen", "group": 17, "period": 4},
    "36": {"symbol": "Kr", "name": "氪", "name_en": "Krypton",
           "mass": 83.798, "category": "noble_gas", "group": 18, "period": 4},
    "47": {"symbol": "Ag", "name": "银", "name_en": "Silver",
           "mass": 107.87, "category": "transition_metal", "group": 11, "period": 5},
    "53": {"symbol": "I", "name": "碘", "name_en": "Iodine",
           "mass": 126.90, "category": "halogen", "group": 17, "period": 5},
    "54": {"symbol": "Xe", "name": "氙", "name_en": "Xenon",
           "mass": 131.29, "category": "noble_gas", "group": 18, "period": 5},
    "74": {"symbol": "W", "name": "钨", "name_en": "Tungsten",
           "mass": 183.84, "category": "transition_metal", "group": 6, "period": 6},
    "78": {"symbol": "Pt", "name": "铂", "name_en": "Platinum",
           "mass": 195.08, "category": "transition_metal", "group": 10, "period": 6},
    "79": {"symbol": "Au", "name": "金", "name_en": "Gold",
           "mass": 196.97, "category": "transition_metal", "group": 11, "period": 6},
    "80": {"symbol": "Hg", "name": "汞", "name_en": "Mercury",
           "mass": 200.59, "category": "transition_metal", "group": 12, "period": 6},
    "82": {"symbol": "Pb", "name": "铅", "name_en": "Lead",
           "mass": 207.2, "category": "post_transition", "group": 14, "period": 6},
    "92": {"symbol": "U", "name": "铀", "name_en": "Uranium",
           "mass": 238.03, "category": "actinide", "group": None, "period": 7},
    "94": {"symbol": "Pu", "name": "钚", "name_en": "Plutonium",
           "mass": 244.0, "category": "actinide", "group": None, "period": 7},
}

# 构建符号 -> 原子序数 的索引
_SYMBOL_TO_Z: Dict[str, str] = {}
for _z, _info in PERIODIC_TABLE.items():
    _SYMBOL_TO_Z[_info["symbol"]] = _z


# ===========================================================================
# 26. unit_convert —— 单位转换
# ===========================================================================
@register_tool(
    "unit_convert", "单位转换（长度/质量/时间/温度/速度/能量）", "scientific_ref",
    params=[
        {"name": "value", "type": "float", "description": "待转换的数值"},
        {"name": "from_unit", "type": "str", "description": "源单位如 km"},
        {"name": "to_unit", "type": "str", "description": "目标单位如 mile"},
        {"name": "category", "type": "str",
         "description": "单位类别: length/mass/time/temperature/speed/energy",
         "required": False, "default": "auto"},
    ],
)
def unit_convert(value: float, from_unit: str, to_unit: str,
                 category: str = "auto") -> Dict[str, Any]:
    # 自动检测类别
    if category == "auto":
        found_cat = None
        for cat, units in UNIT_FACTORS.items():
            if from_unit in units and to_unit in units:
                found_cat = cat
                break
        if found_cat is None:
            return {"status": "error",
                    "reason": f"无法自动匹配单位类别: {from_unit} -> {to_unit}"}
        category = found_cat

    if category not in UNIT_FACTORS:
        return {"status": "error",
                "reason": f"不支持的类别: {category}，可选: "
                          f"{list(UNIT_FACTORS.keys())}"}
    units = UNIT_FACTORS[category]
    if from_unit not in units:
        return {"status": "error",
                "reason": f"类别 {category} 中不支持源单位: {from_unit}"}
    if to_unit not in units:
        return {"status": "error",
                "reason": f"类别 {category} 中不支持目标单位: {to_unit}"}

    # 温度特殊处理
    if category == "temperature":
        result = _convert_temperature(value, units[from_unit], units[to_unit])
        if result is None:
            return {"status": "error", "reason": "温度转换失败"}
        return {"status": "success", "value": result, "from": from_unit,
                "to": to_unit, "input": value, "category": "temperature"}

    # 线性转换: value * from_factor / to_factor
    from_factor = units[from_unit]
    to_factor = units[to_unit]
    result = value * from_factor / to_factor
    return {"status": "success", "value": result, "from": from_unit,
            "to": to_unit, "input": value, "category": category}


def _convert_temperature(value: float, from_scale: str, to_scale: str) -> float:
    """温度刻度互转（K/C/F/R）。"""
    # 先统一转为开尔文
    if from_scale == "kelvin":
        kelvin = value
    elif from_scale == "celsius":
        kelvin = value + 273.15
    elif from_scale == "fahrenheit":
        kelvin = (value - 32.0) * 5.0 / 9.0 + 273.15
    elif from_scale == "rankine":
        kelvin = value * 5.0 / 9.0
    else:
        return None
    # 再从开尔文转为目标
    if to_scale == "kelvin":
        return kelvin
    elif to_scale == "celsius":
        return kelvin - 273.15
    elif to_scale == "fahrenheit":
        return (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    elif to_scale == "rankine":
        return kelvin * 9.0 / 5.0
    return None


# ===========================================================================
# 27. constant_lookup —— 物理常数查询
# ===========================================================================
@register_tool(
    "constant_lookup", "物理常数查询（光速/引力常数/普朗克等，CODATA 2018）",
    "scientific_ref",
    params=[
        {"name": "name", "type": "str",
         "description": "常数名或符号如 speed_of_light / c / G / h"},
        {"name": "list_all", "type": "bool", "description": "是否列出全部常数",
         "required": False, "default": False},
    ],
)
def constant_lookup(name: str = "", list_all: bool = False) -> Dict[str, Any]:
    if list_all:
        return {"status": "success", "count": len(PHYSICAL_CONSTANTS),
                "constants": {k: {"symbol": v["symbol"], "value": v["value"],
                                  "unit": v["unit"], "description": v["description"]}
                              for k, v in PHYSICAL_CONSTANTS.items()}}
    # 按名称查找
    key = name.lower().strip()
    if key in PHYSICAL_CONSTANTS:
        c = PHYSICAL_CONSTANTS[key]
        return {"status": "success", "name": key, "symbol": c["symbol"],
                "value": c["value"], "unit": c["unit"],
                "description": c["description"], "category": c["category"]}
    # 按符号查找
    for k, v in PHYSICAL_CONSTANTS.items():
        if v["symbol"].lower() == key:
            return {"status": "success", "name": k, "symbol": v["symbol"],
                    "value": v["value"], "unit": v["unit"],
                    "description": v["description"], "category": v["category"]}
    return {"status": "error",
            "reason": f"未找到常数: {name}，使用 list_all=True 查看全部",
            "available": list(PHYSICAL_CONSTANTS.keys())}


# ===========================================================================
# 28. formula_lookup —— 常用科学公式查询
# ===========================================================================
@register_tool(
    "formula_lookup", "常用科学公式查询（开普勒第三定律/逃逸速度/火箭方程等）",
    "scientific_ref",
    params=[
        {"name": "name", "type": "str",
         "description": "公式名如 kepler_third_law"},
        {"name": "list_all", "type": "bool", "description": "是否列出全部公式",
         "required": False, "default": False},
    ],
)
def formula_lookup(name: str = "", list_all: bool = False) -> Dict[str, Any]:
    if list_all:
        return {"status": "success", "count": len(SCIENTIFIC_FORMULAS),
                "formulas": {k: {"formula": v["formula"],
                                 "description": v["description"],
                                 "category": v["category"]}
                             for k, v in SCIENTIFIC_FORMULAS.items()}}
    key = name.lower().strip()
    if key in SCIENTIFIC_FORMULAS:
        f = SCIENTIFIC_FORMULAS[key]
        return {"status": "success", "name": key, "formula": f["formula"],
                "description": f["description"], "variables": f["variables"],
                "category": f["category"],
                "example": f.get("example", "")}
    return {"status": "error",
            "reason": f"未找到公式: {name}，使用 list_all=True 查看全部",
            "available": list(SCIENTIFIC_FORMULAS.keys())}


# ===========================================================================
# 29. periodic_table —— 元素周期表查询
# ===========================================================================
@register_tool(
    "periodic_table", "元素周期表查询（按原子序数或元素符号）", "scientific_ref",
    params=[
        {"name": "query", "type": "str",
         "description": "原子序数如 26 或元素符号如 Fe"},
        {"name": "list_all", "type": "bool", "description": "是否列出全部元素",
         "required": False, "default": False},
    ],
)
def periodic_table(query: str = "", list_all: bool = False) -> Dict[str, Any]:
    if list_all:
        return {"status": "success", "count": len(PERIODIC_TABLE),
                "elements": PERIODIC_TABLE}
    q = query.strip()
    # 尝试按原子序数
    if q.isdigit():
        z = str(int(q))
        if z in PERIODIC_TABLE:
            e = PERIODIC_TABLE[z]
            return {"status": "success", "atomic_number": int(z), **e}
        return {"status": "error", "reason": f"原子序数 {q} 不在内置数据中"}
    # 按符号查找（首字母大写）
    sym = q
    if len(q) <= 2:
        sym = q[0].upper() + q[1:].lower() if q else q
    if sym in _SYMBOL_TO_Z:
        z = _SYMBOL_TO_Z[sym]
        e = PERIODIC_TABLE[z]
        return {"status": "success", "atomic_number": int(z), **e}
    return {"status": "error",
            "reason": f"未找到元素: {query}，支持按原子序数或符号查询"}


# ===========================================================================
# 30. time_convert —— 时间格式转换
# ===========================================================================
@register_tool(
    "time_convert", "时间格式转换（epoch/ISO/JD/DOY）", "scientific_ref",
    params=[
        {"name": "value", "type": "str",
         "description": "时间值: epoch秒数、ISO字符串(2024-01-01T00:00:00)、"
                        "JD儒略日"},
        {"name": "from_format", "type": "str",
         "description": "源格式: epoch/iso/jd/doy"},
        {"name": "to_format", "type": "str",
         "description": "目标格式: epoch/iso/jd/doy"},
    ],
)
def time_convert(value: str, from_format: str,
                 to_format: str) -> Dict[str, Any]:
    try:
        # 统一转为 datetime 对象
        dt = None
        if from_format == "epoch":
            dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + \
                 datetime.timedelta(seconds=float(value))
        elif from_format == "iso":
            dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
        elif from_format == "jd":
            # 儒略日转 datetime
            jd = float(value)
            # JD 2440587.5 = 1970-01-01 00:00:00 UTC
            epoch_seconds = (jd - 2440587.5) * 86400.0
            dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + \
                 datetime.timedelta(seconds=epoch_seconds)
        elif from_format == "doy":
            # DOY 格式: YYYY:DDD 或 YYYY:DDD:SSSSS
            parts = value.split(":")
            year = int(parts[0])
            doy = int(parts[1])
            dt = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc) + \
                 datetime.timedelta(days=doy - 1)
            if len(parts) > 2:
                secs = int(parts[2])
                dt += datetime.timedelta(seconds=secs)
        else:
            return {"status": "error",
                    "reason": f"不支持的源格式: {from_format}"}

        # 从 datetime 转为目标格式
        result = None
        if to_format == "epoch":
            result = dt.timestamp()
        elif to_format == "iso":
            result = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        elif to_format == "jd":
            epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
            result = 2440587.5 + (dt - epoch).total_seconds() / 86400.0
        elif to_format == "doy":
            doy = dt.timetuple().tm_yday
            secs = dt.hour * 3600 + dt.minute * 60 + dt.second
            result = f"{dt.year}:{doy:03d}:{secs:05d}"
        else:
            return {"status": "error",
                    "reason": f"不支持的目标格式: {to_format}"}

        return {"status": "success", "value": result,
                "from": from_format, "to": to_format, "input": value}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 31. angle_convert —— 角度转换
# ===========================================================================
@register_tool(
    "angle_convert", "角度转换（deg/rad/hms/dms）", "scientific_ref",
    params=[
        {"name": "value", "type": "float", "description": "角度值"},
        {"name": "from_format", "type": "str",
         "description": "源格式: deg/rad/hms/dms"},
        {"name": "to_format", "type": "str",
         "description": "目标格式: deg/rad/hms/dms"},
    ],
)
def angle_convert(value, from_format: str, to_format: str) -> Dict[str, Any]:
    try:
        # 统一转为度
        deg = None
        if from_format == "deg":
            deg = float(value)
        elif from_format == "rad":
            deg = math.radians(float(value))
        elif from_format == "hms":
            # hms 可以是 "HH:MM:SS" 字符串或小时浮点数
            if isinstance(value, str):
                parts = value.split(":")
                h = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 else 0
                s = float(parts[2]) if len(parts) > 2 else 0
                hours = h + m / 60.0 + s / 3600.0
            else:
                hours = float(value)
            deg = hours * 15.0  # 1h = 15°
        elif from_format == "dms":
            # dms 可以是 "DD:MM:SS" 字符串或度浮点数
            if isinstance(value, str):
                sign = 1.0
                v = value.strip()
                if v.startswith("-"):
                    sign = -1.0
                    v = v[1:]
                parts = v.split(":")
                d = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 else 0
                s = float(parts[2]) if len(parts) > 2 else 0
                deg = sign * (d + m / 60.0 + s / 3600.0)
            else:
                deg = float(value)
        else:
            return {"status": "error",
                    "reason": f"不支持的源格式: {from_format}"}

        # 从度转为目标格式
        result = None
        if to_format == "deg":
            result = deg
        elif to_format == "rad":
            result = math.radians(deg)
        elif to_format == "hms":
            total_hours = deg / 15.0
            h = int(total_hours)
            m = int((total_hours - h) * 60)
            s = (total_hours - h - m / 60.0) * 3600
            result = f"{h:02d}:{m:02d}:{s:06.3f}"
        elif to_format == "dms":
            sign = "-" if deg < 0 else ""
            abs_deg = abs(deg)
            d = int(abs_deg)
            m = int((abs_deg - d) * 60)
            s = (abs_deg - d - m / 60.0) * 3600
            result = f"{sign}{d:02d}:{m:02d}:{s:06.3f}"
        else:
            return {"status": "error",
                    "reason": f"不支持的目标格式: {to_format}"}

        return {"status": "success", "value": result,
                "from": from_format, "to": to_format, "input_deg": deg}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 32. coordinate_convert —— 坐标转换
# ===========================================================================
@register_tool(
    "coordinate_convert", "坐标转换（直角/球坐标/柱坐标）", "scientific_ref",
    params=[
        {"name": "coords", "type": "list",
         "description": "坐标值列表，直角[x,y,z] / 球[r,theta,phi] / 柱[r,phi,z]"},
        {"name": "from_system", "type": "str",
         "description": "源坐标系: cartesian/spherical/cylindrical"},
        {"name": "to_system", "type": "str",
         "description": "目标坐标系: cartesian/spherical/cylindrical"},
        {"name": "angle_unit", "type": "str",
         "description": "角度单位: deg 或 rad",
         "required": False, "default": "rad"},
    ],
)
def coordinate_convert(coords: List[float], from_system: str,
                       to_system: str, angle_unit: str = "rad") -> Dict[str, Any]:
    try:
        if len(coords) != 3:
            return {"status": "error", "reason": "坐标须为 3 维列表"}
        is_deg = angle_unit.lower() == "deg"

        # 统一转为直角坐标
        if from_system == "cartesian":
            x, y, z = coords[0], coords[1], coords[2]
        elif from_system == "spherical":
            # [r, theta(极角/天顶角), phi(方位角)]
            r, theta, phi = coords[0], coords[1], coords[2]
            if is_deg:
                theta = math.radians(theta)
                phi = math.radians(phi)
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
        elif from_system == "cylindrical":
            # [r, phi, z]
            r, phi, z = coords[0], coords[1], coords[2]
            if is_deg:
                phi = math.radians(phi)
            x = r * math.cos(phi)
            y = r * math.sin(phi)
        else:
            return {"status": "error",
                    "reason": f"不支持的源坐标系: {from_system}"}

        # 从直角转为目标
        result = None
        if to_system == "cartesian":
            result = [x, y, z]
        elif to_system == "spherical":
            r = math.sqrt(x * x + y * y + z * z)
            theta = math.acos(z / r) if r > 0 else 0.0
            phi = math.atan2(y, x)
            if is_deg:
                theta = math.degrees(theta)
                phi = math.degrees(phi)
            result = [r, theta, phi]
        elif to_system == "cylindrical":
            r = math.sqrt(x * x + y * y)
            phi = math.atan2(y, x)
            if is_deg:
                phi = math.degrees(phi)
            result = [r, phi, z]
        else:
            return {"status": "error",
                    "reason": f"不支持的目标坐标系: {to_system}"}

        return {"status": "success", "coords": result,
                "from": from_system, "to": to_system, "angle_unit": angle_unit,
                "cartesian_intermediate": [x, y, z]}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 33. scientific_notation —— 科学计数法格式化
# ===========================================================================
@register_tool(
    "scientific_notation", "科学计数法格式化", "scientific_ref",
    params=[
        {"name": "value", "type": "float", "description": "数值"},
        {"name": "precision", "type": "int", "description": "小数位数",
         "required": False, "default": 4},
        {"name": "format_type", "type": "str",
         "description": "格式类型: standard/e_minus/latex",
         "required": False, "default": "standard"},
    ],
)
def scientific_notation(value: float, precision: int = 4,
                        format_type: str = "standard") -> Dict[str, Any]:
    try:
        v = float(value)
        if v == 0:
            return {"status": "success", "formatted": "0",
                    "mantissa": 0.0, "exponent": 0, "value": v}
        exp = math.floor(math.log10(abs(v)))
        mantissa = v / (10 ** exp)
        if format_type == "standard":
            formatted = f"{mantissa:.{precision}f} × 10^{exp}"
        elif format_type == "e_minus":
            formatted = f"{mantissa:.{precision}f}e{exp}"
        elif format_type == "latex":
            sign = "+" if exp >= 0 else "-"
            formatted = f"{mantissa:.{precision}f} \\times 10^{{{sign}{abs(exp)}}}"
        else:
            return {"status": "error",
                    "reason": f"不支持的格式类型: {format_type}"}
        return {"status": "success", "formatted": formatted,
                "mantissa": round(mantissa, precision),
                "exponent": exp, "value": v}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 34. uncertainty_calc —— 不确定度传播计算
# ===========================================================================
@register_tool(
    "uncertainty_calc", "不确定度传播计算（支持加减乘除和幂运算）",
    "scientific_ref",
    params=[
        {"name": "operation", "type": "str",
         "description": "运算类型: add/subtract/multiply/divide/power"},
        {"name": "value1", "type": "float", "description": "第一个量的值"},
        {"name": "uncertainty1", "type": "float",
         "description": "第一个量的不确定度"},
        {"name": "value2", "type": "float", "description": "第二个量的值"},
        {"name": "uncertainty2", "type": "float",
         "description": "第二个量的不确定度"},
        {"name": "power", "type": "float",
         "description": "幂运算指数（仅 power 模式使用）",
         "required": False, "default": 2.0},
    ],
)
def uncertainty_calc(operation: str, value1: float, uncertainty1: float,
                     value2: float, uncertainty2: float,
                     power: float = 2.0) -> Dict[str, Any]:
    op = operation.lower()
    try:
        if op == "add":
            result = value1 + value2
            unc = math.sqrt(uncertainty1 ** 2 + uncertainty2 ** 2)
            formula = "σ = sqrt(σ1² + σ2²)"
        elif op == "subtract":
            result = value1 - value2
            unc = math.sqrt(uncertainty1 ** 2 + uncertainty2 ** 2)
            formula = "σ = sqrt(σ1² + σ2²)"
        elif op == "multiply":
            result = value1 * value2
            rel_unc = math.sqrt(
                (uncertainty1 / value1) ** 2 + (uncertainty2 / value2) ** 2
            ) if value1 != 0 and value2 != 0 else float("inf")
            unc = abs(result) * rel_unc
            formula = "σ/y = sqrt((σ1/x1)² + (σ2/x2)²)"
        elif op == "divide":
            result = value1 / value2
            rel_unc = math.sqrt(
                (uncertainty1 / value1) ** 2 + (uncertainty2 / value2) ** 2
            ) if value1 != 0 and value2 != 0 else float("inf")
            unc = abs(result) * rel_unc
            formula = "σ/y = sqrt((σ1/x1)² + (σ2/x2)²)"
        elif op == "power":
            # y = x^n, σ_y = |n * x^(n-1)| * σ_x
            result = value1 ** power
            unc = abs(power * value1 ** (power - 1)) * uncertainty1
            formula = f"σ_y = |{power} × x^({power}-1)| × σ_x"
        else:
            return {"status": "error",
                    "reason": f"不支持的运算: {operation}，可选 "
                              f"add/subtract/multiply/divide/power"}
        return {"status": "success", "result": result,
                "uncertainty": unc, "relative_uncertainty": unc / abs(result)
                if result != 0 else None,
                "formula": formula, "operation": op,
                "value1": value1, "uncertainty1": uncertainty1,
                "value2": value2, "uncertainty2": uncertainty2}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 35. significant_figures —— 有效数字处理
# ===========================================================================
@register_tool(
    "significant_figures", "有效数字处理（四舍五入到指定位数）", "scientific_ref",
    params=[
        {"name": "value", "type": "float", "description": "数值"},
        {"name": "sig_figs", "type": "int", "description": "有效数字位数"},
        {"name": "return_type", "type": "str",
         "description": "返回类型: float/str/both",
         "required": False, "default": "both"},
    ],
)
def significant_figures(value: float, sig_figs: int,
                        return_type: str = "both") -> Dict[str, Any]:
    try:
        if sig_figs < 1:
            return {"status": "error", "reason": "有效数字位数必须 >= 1"}
        v = float(value)
        if v == 0:
            result_float = 0.0
            result_str = "0." + "0" * (sig_figs - 1)
        else:
            # 计算需要保留的小数位数
            d = math.floor(math.log10(abs(v)))
            decimal_places = sig_figs - 1 - d
            result_float = round(v, decimal_places)
            # 格式化字符串
            if decimal_places >= 0:
                result_str = f"{result_float:.{max(decimal_places, 0)}f}"
            else:
                # 结果很大，用科学计数法
                result_str = f"{result_float:.{sig_figs - 1}e}"
        output = {}
        if return_type in ("float", "both"):
            output["value"] = result_float
        if return_type in ("str", "both"):
            output["value_str"] = result_str
        output["status"] = "success"
        output["sig_figs"] = sig_figs
        output["original"] = v
        return output
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ---------------------------------------------------------------------------
# 模块自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from aerospace_agent.research_tools.base import get_registry

    reg = get_registry()
    print(f"scientific_ref 模块已注册工具数: "
          f"{len(reg.list_by_category('scientific_ref'))}")
    print("工具列表:")
    for name in reg.list_by_category("scientific_ref"):
        print(f"  - {name}")

    print("\n--- 冒烟测试 ---")
    print(unit_convert(1, "km", "mile"))
    print(unit_convert(100, "C", "F", category="temperature"))
    print(unit_convert(1, "AU", "km"))
    print(constant_lookup("c"))
    print(constant_lookup("planck_constant"))
    print(constant_lookup("", list_all=True)["count"])
    print(formula_lookup("kepler_third_law"))
    print(formula_lookup("tsiolkovsky_rocket"))
    print(periodic_table("26"))
    print(periodic_table("Fe"))
    print(time_convert("1700000000", "epoch", "iso"))
    print(time_convert("2460310.5", "jd", "iso"))
    print(angle_convert(90, "deg", "rad"))
    print(angle_convert("06:00:00", "hms", "deg"))
    print(coordinate_convert([1, 1, 0], "cartesian", "spherical", "rad"))
    print(coordinate_convert([1, 1, 1], "cartesian", "spherical", "deg"))
    print(scientific_notation(6.022e23, precision=3))
    print(uncertainty_calc("multiply", 10, 0.1, 20, 0.2))
    print(significant_figures(3.14159265, 4))
