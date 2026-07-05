"""公式渲染辅助 (LaTeX formula registry for reports)。

本模块集中存放地月转移轨道设计所涉及的关键物理公式的 LaTeX 字符串与
推导步骤文本, 供 ``report.py`` 生成 MathJax 渲染的 HTML 报告, 以及
``generate_summary_text`` 生成纯文本摘要时引用。

公式命名 (FORMULAS dict 的 key)
-------------------------------
vis_viva               Vis-viva 方程 (能量守恒 -> 速度)
kepler_third           开普勒第三定律 (周期-半长轴)
hohmann_dv1            Hohmann 转移第一脉冲 (离轨加速)
hohmann_dv2            Hohmann 转移第二脉冲 (入轨制动)
hohmann_transfer_time  Hohmann 转移飞行时间 (半周期)
phase_angle            发射窗口相位角条件
C3                     特征能量 C3 (双曲剩余速度平方)
specific_energy        比能量 (轨道能量)
soi                    引力作用球半径 (Laplace)
perilune_velocity      双曲线近月点速度
circular_velocity      圆轨道速度
escape_velocity        逃逸速度
eccentricity           偏心率 (由近/远地点)
delta_v_capture        捕获 Δv (双曲线 -> 圆轨道)

LaTeX 字符串使用 raw string (r'...'), 由 MathJax 在浏览器端渲染。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 公式集合 (LaTeX, raw string)
# ---------------------------------------------------------------------------
FORMULAS: dict[str, str] = {
    # Vis-viva 方程: 由能量守恒推出任意点速度
    "vis_viva": r"v^2 = \mu\left(\dfrac{2}{r} - \dfrac{1}{a}\right)",

    # 开普勒第三定律: 周期与半长轴关系
    "kepler_third": r"T^2 = \dfrac{4\pi^2}{\mu}\, a^3",

    # Hohmann 转移第一脉冲 (LEO 离轨, 切向加速)
    "hohmann_dv1": (
        r"\Delta v_1 = "
        r"\sqrt{\mu\left(\dfrac{2}{r_1} - \dfrac{1}{a_t}\right)} "
        r"- \sqrt{\dfrac{\mu}{r_1}}"
    ),

    # Hohmann 转移第二脉冲 (远地点入轨制动)
    "hohmann_dv2": (
        r"\Delta v_2 = "
        r"\sqrt{\dfrac{\mu}{r_2}} "
        r"- \sqrt{\mu\left(\dfrac{2}{r_2} - \dfrac{1}{a_t}\right)}"
    ),

    # Hohmann 转移椭圆半长轴
    "hohmann_semi_major": r"a_t = \dfrac{r_1 + r_2}{2}",

    # Hohmann 转移飞行时间 = 半周期
    "hohmann_transfer_time": r"t_f = \pi\sqrt{\dfrac{a_t^3}{\mu}}",

    # 发射窗口相位角条件 (月球须超前出发方向)
    "phase_angle": r"\phi_{\mathrm{req}} = \pi - \omega_m\, t_f",

    # 特征能量 C3 (发射能量指标)
    "C3": r"C_3 = v_\infty^2 = v^2 - \dfrac{2\mu}{r}",

    # 比能量 (轨道能量, 椭圆 a>0 则 eps<0)
    "specific_energy": r"\varepsilon = \dfrac{v^2}{2} - \dfrac{\mu}{r} = -\dfrac{\mu}{2a}",

    # 引力作用球半径 (Laplace)
    "soi": r"r_{\mathrm{SOI}} = a\left(\dfrac{m_2}{m_1}\right)^{2/5} = a\left(\dfrac{\mu_2}{\mu_1}\right)^{2/5}",

    # 双曲线近月点速度 (vis-viva, a<0)
    "perilune_velocity": r"v_p^2 = v_\infty^2 + \dfrac{2\mu_m}{r_p}",

    # 圆轨道速度
    "circular_velocity": r"v_{\mathrm{circ}} = \sqrt{\dfrac{\mu}{r}}",

    # 逃逸速度
    "escape_velocity": r"v_{\mathrm{esc}} = \sqrt{\dfrac{2\mu}{r}}",

    # 偏心率 (由近地点/远地点)
    "eccentricity": r"e = \dfrac{r_a - r_p}{r_a + r_p}",

    # 捕获 Δv (双曲线近月点 -> 圆轨道)
    "delta_v_capture": (
        r"\Delta v_{\mathrm{LOI}} = "
        r"\sqrt{v_\infty^2 + \dfrac{2\mu_m}{r_p}} "
        r"- \sqrt{\dfrac{\mu_m}{r_p}}"
    ),

    # 总 Δv
    "delta_v_total": r"\Delta v_{\mathrm{total}} = \Delta v_1 + \Delta v_2",
}

# ---------------------------------------------------------------------------
# 公式推导步骤 (多行文本, 每步一行)
# ---------------------------------------------------------------------------
DERIVATIONS: dict[str, str] = {
    "vis_viva": (
        "Vis-viva 方程推导 (能量守恒):\n"
        "1. 比能量 (单位质量能量): ε = v²/2 − μ/r\n"
        "2. 对椭圆轨道, ε = −μ/(2a) (a 为半长轴)\n"
        "3. 联立: v²/2 − μ/r = −μ/(2a)\n"
        "4. 解出 v²: v² = μ(2/r − 1/a)  ∎"
    ),
    "kepler_third": (
        "开普勒第三定律推导:\n"
        "1. 圆轨道: 引力 = 向心力  μ/r² = ω²r,  ω = 2π/T\n"
        "2. 故 (2π/T)² = μ/r³  =>  T² = (4π²/μ) r³\n"
        "3. 对椭圆轨道, r → a (半长轴), 同样成立:\n"
        "   T² = (4π²/μ) a³  ∎"
    ),
    "hohmann_dv1": (
        "Hohmann 第一脉冲 (TLI) 推导:\n"
        "1. 转移椭圆半长轴: a_t = (r₁ + r₂)/2\n"
        "2. LEO 圆轨道速度: v_circ₁ = √(μ/r₁)\n"
        "3. 转移椭圆近地点速度 (vis-viva, r=r₁):\n"
        "   v_per = √(μ(2/r₁ − 1/a_t))\n"
        "4. 切向加速脉冲: Δv₁ = v_per − v_circ₁  ∎"
    ),
    "hohmann_dv2": (
        "Hohmann 第二脉冲 (LOI) 推导:\n"
        "1. 转移椭圆远地点速度 (vis-viva, r=r₂):\n"
        "   v_apo = √(μ(2/r₂ − 1/a_t))\n"
        "2. 目标圆轨道速度: v_circ₂ = √(μ/r₂)\n"
        "3. 切向制动脉冲: Δv₂ = v_circ₂ − v_apo  ∎"
    ),
    "hohmann_transfer_time": (
        "Hohmann 转移飞行时间推导:\n"
        "1. 转移椭圆完整周期 (开普勒第三定律):\n"
        "   T_t = 2π√(a_t³/μ)\n"
        "2. Hohmann 转移只走半个椭圆 (近地点→远地点):\n"
        "   t_f = T_t/2 = π√(a_t³/μ)  ∎"
    ),
    "phase_angle": (
        "发射窗口相位角推导:\n"
        "1. 约定: 停泊轨道近地点 (出发方向) 固定在惯性 +x\n"
        "2. Hohmann 转移远地点在 −x (航天器经 t_f 到达)\n"
        "3. 月球须在 t_f 后也到达 −x (角速度 ω_m, 顺行)\n"
        "4. 故发射时刻月球应位于: φ_req = π − ω_m·t_f\n"
        "   (从 +x 起算, 顺行方向; 月球须 *超前* 该角度)  ∎"
    ),
    "C3": (
        "特征能量 C3 推导:\n"
        "1. 比能量: ε = v²/2 − μ/r\n"
        "2. 双曲剩余速度 (无穷远处): v∞² = 2ε = v² − 2μ/r\n"
        "3. C3 即 v∞², 表征逃逸所需能量 (C3>0 为双曲):\n"
        "   C3 = v∞² = v² − 2μ/r  ∎"
    ),
    "specific_energy": (
        "比能量推导:\n"
        "1. 动能 + 势能 (单位质量): ε = v²/2 − μ/r\n"
        "2. 由 vis-viva 代入 v²: ε = μ(2/r−1/a)/2 − μ/r = −μ/(2a)\n"
        "3. 椭圆 a>0 ⇒ ε<0 (束缚); 双曲 a<0 ⇒ ε>0 (逃逸);\n"
        "   抛物 a→∞ ⇒ ε=0 (临界)  ∎"
    ),
    "soi": (
        "引力作用球 (SOI) 半径推导 (Laplace):\n"
        "1. 限制性三体问题: 比较次天体引力 vs 主天体潮汐加速度\n"
        "2. 次天体引力 ~ μ₂/r²; 主天体差分 ~ μ₁·r/R³\n"
        "3. 令两者相等: μ₂/r² = μ₁·r/R³  ⇒  r² = (μ₂/μ₁)·R³/r\n"
        "4. 取 r ~ R (轨道半径 a), 解出 SOI 边界:\n"
        "   r_SOI = a·(μ₂/μ₁)^(2/5)  ∎"
    ),
    "perilune_velocity": (
        "双曲线近月点速度推导:\n"
        "1. 月心双曲线: a < 0, v∞² = −μ_m/a (能量)\n"
        "2. vis-viva: v² = μ_m(2/r − 1/a) = μ_m·2/r + v∞²\n"
        "3. 近月点 r = r_p: v_p = √(v∞² + 2μ_m/r_p)  ∎"
    ),
    "delta_v_capture": (
        "捕获 Δv (LOI) 推导:\n"
        "1. 双曲线近月点速度: v_p = √(v∞² + 2μ_m/r_p)\n"
        "2. 目标近月圆轨道速度: v_circ = √(μ_m/r_p)\n"
        "3. 单次脉冲制动: Δv_LOI = v_p − v_circ  ∎"
    ),
}


def get_formula_latex(name: str) -> str:
    """按名称取 LaTeX 公式字符串。

    参数
    -----
    name : 公式名 (见 FORMULAS 的 key)

    返回
    -----
    LaTeX 字符串; 若名称不存在, 返回占位提示。
    """
    if name not in FORMULAS:
        return r"\text{(formula '%s' not found)}" % name
    return FORMULAS[name]


def get_derivation(name: str) -> str:
    """按名称取公式推导步骤文本 (纯文本, 多行)。

    参数
    -----
    name : 公式名 (见 DERIVATIONS 的 key)

    返回
    -----
    推导步骤文本; 若名称不存在, 返回占位提示。
    """
    if name not in DERIVATIONS:
        return f"(derivation for '{name}' not found)"
    return DERIVATIONS[name]


def list_formulas() -> list[str]:
    """返回所有可用公式名列表。"""
    return list(FORMULAS.keys())


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.reporting.formulas 自测 ===")
    print(f"公式总数: {len(FORMULAS)}")
    print(f"推导总数: {len(DERIVATIONS)}")
    for name in ["vis_viva", "kepler_third", "hohmann_dv1", "C3", "soi", "phase_angle"]:
        latex = get_formula_latex(name)
        print(f"\n[{name}]")
        print(f"  LaTeX: {latex}")
        print(f"  推导:\n    {get_derivation(name).replace(chr(10), chr(10)+'    ')}")
    # 不存在的公式
    assert "not found" in get_formula_latex("nonexistent")
    assert "not found" in get_derivation("nonexistent")
    assert set(FORMULAS.keys()) == set(DERIVATIONS.keys()) or \
           set(DERIVATIONS.keys()).issubset(set(FORMULAS.keys()))
    print("\nformulas 自测通过.")
