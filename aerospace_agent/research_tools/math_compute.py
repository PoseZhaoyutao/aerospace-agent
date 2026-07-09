"""数学与计算工具集——15 个纯 Python 实现的科研计算原子工具。

第一性原理：
  1. 所有数值算法仅依赖标准库 ``math`` / ``random``，不强制 numpy/scipy
  2. 每个工具是不可分解的原子操作，输入/输出均为 JSON 可序列化
  3. 涉及函数表达式求值的工具使用受限 ``eval``（禁用内置函数，仅开放 math）
  4. 矩阵运算全部以嵌套 list 表示，纯 Python 实现
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List

from aerospace_agent.research_tools.base import register_tool

# ---------------------------------------------------------------------------
# 安全 eval 辅助：受限命名空间，仅允许 math 模块中的数学函数
# ---------------------------------------------------------------------------
_SAFE_NAMES: Dict[str, Any] = {
    "math": math,
    # 常用数学函数直接暴露，方便书写表达式
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh,
    "tanh": math.tanh, "exp": math.exp, "log": math.log,
    "log10": math.log10, "log2": math.log2, "sqrt": math.sqrt,
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
    "pow": pow, "abs": abs,
}


def _safe_eval(expression: str, x: float = 0.0) -> float:
    """在受限命名空间中求值表达式，变量 ``x`` 可用。"""
    local_ns: Dict[str, Any] = dict(_SAFE_NAMES)
    local_ns["x"] = x
    return float(eval(expression, {"__builtins__": {}}, local_ns))


# ===========================================================================
# 1. basic_arithmetic —— 基本四则运算
# ===========================================================================
@register_tool(
    "basic_arithmetic", "基本四则运算（安全 eval，支持 math 函数）", "math_compute",
    params=[
        {"name": "expression", "type": "str",
         "description": "数学表达式如 1+2*3、sin(pi/2)、sqrt(16)"},
    ],
)
def basic_arithmetic(expression: str) -> Dict[str, Any]:
    try:
        result = eval(expression, {"__builtins__": {}}, dict(_SAFE_NAMES))
        return {"status": "success", "expression": expression, "result": result}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 2. matrix_multiply —— 矩阵乘法（纯 Python）
# ===========================================================================
@register_tool(
    "matrix_multiply", "矩阵乘法 A×B（纯 Python 实现）", "math_compute",
    params=[
        {"name": "matrix_a", "type": "list",
         "description": "矩阵 A，二维 list 如 [[1,2],[3,4]]"},
        {"name": "matrix_b", "type": "list",
         "description": "矩阵 B，二维 list 如 [[5,6],[7,8]]"},
    ],
)
def matrix_multiply(matrix_a: List[List[float]],
                    matrix_b: List[List[float]]) -> Dict[str, Any]:
    if not matrix_a or not matrix_b or not matrix_a[0] or not matrix_b[0]:
        return {"status": "error", "reason": "矩阵不能为空"}
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])
    if cols_a != rows_b:
        return {"status": "error",
                "reason": f"维度不匹配: A 是 {rows_a}x{cols_a}, B 是 {rows_b}x{cols_b}"}
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = s
    return {"status": "success", "result": result,
            "shape": [rows_a, cols_b]}


# ===========================================================================
# 3. matrix_invert —— 矩阵求逆（高斯-约旦消元法）
# ===========================================================================
@register_tool(
    "matrix_invert", "矩阵求逆（纯 Python 高斯-约旦消元法）", "math_compute",
    params=[
        {"name": "matrix", "type": "list",
         "description": "方阵，二维 list 如 [[4,7],[2,6]]"},
    ],
)
def matrix_invert(matrix: List[List[float]]) -> Dict[str, Any]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        return {"status": "error", "reason": "输入必须是方阵"}
    # 构造增广矩阵 [A | I]
    aug = [list(matrix[i]) + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    for col in range(n):
        # 选主元
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            return {"status": "error", "reason": "矩阵奇异（不可逆）"}
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        # 归一化主元行
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        # 消去其它行
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]
    inverse = [row[n:] for row in aug]
    return {"status": "success", "inverse": inverse, "size": n}


# ===========================================================================
# 4. linear_solve —— 解线性方程组 Ax = b
# ===========================================================================
@register_tool(
    "linear_solve", "解线性方程组 Ax=b（高斯消元法）", "math_compute",
    params=[
        {"name": "matrix_a", "type": "list",
         "description": "系数矩阵 A，二维 list"},
        {"name": "vector_b", "type": "list",
         "description": "右端向量 b，一维 list"},
    ],
)
def linear_solve(matrix_a: List[List[float]],
                 vector_b: List[float]) -> Dict[str, Any]:
    n = len(matrix_a)
    if n == 0 or any(len(row) != n for row in matrix_a) or len(vector_b) != n:
        return {"status": "error", "reason": "A 必须为 n×n 方阵且 b 长度为 n"}
    # 增广矩阵
    aug = [list(matrix_a[i]) + [vector_b[i]] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            return {"status": "error", "reason": "矩阵奇异，方程组无唯一解"}
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(n + 1)]
    solution = [aug[i][n] for i in range(n)]
    return {"status": "success", "x": solution, "size": n}


# ===========================================================================
# 5. root_find —— 方程求根（二分法）
# ===========================================================================
@register_tool(
    "root_find", "方程求根（二分法），表达式以 x 为变量", "math_compute",
    params=[
        {"name": "expression", "type": "str",
         "description": "函数表达式如 x**3-x-2，变量为 x"},
        {"name": "a", "type": "float", "description": "区间左端点"},
        {"name": "b", "type": "float", "description": "区间右端点"},
        {"name": "tol", "type": "float", "description": "容差",
         "required": False, "default": 1e-10},
        {"name": "max_iter", "type": "int", "description": "最大迭代次数",
         "required": False, "default": 1000},
    ],
)
def root_find(expression: str, a: float, b: float,
              tol: float = 1e-10, max_iter: int = 1000) -> Dict[str, Any]:
    try:
        fa = _safe_eval(expression, a)
        fb = _safe_eval(expression, b)
    except Exception as e:
        return {"status": "error", "reason": f"表达式求值失败: {e}"}
    if fa * fb > 0:
        return {"status": "error",
                "reason": f"区间端点同号: f({a})={fa}, f({b})={fb}，无法使用二分法"}
    for i in range(max_iter):
        mid = (a + b) / 2.0
        fm = _safe_eval(expression, mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return {"status": "success", "root": mid, "iterations": i + 1,
                    "f_root": fm}
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return {"status": "error", "reason": f"达到最大迭代次数 {max_iter} 未收敛",
            "best_estimate": (a + b) / 2.0}


# ===========================================================================
# 6. integrate_num —— 数值积分（梯形法）
# ===========================================================================
@register_tool(
    "integrate_num", "数值积分（梯形法），表达式以 x 为变量", "math_compute",
    params=[
        {"name": "expression", "type": "str",
         "description": "被积函数表达式如 sin(x)*x，变量为 x"},
        {"name": "a", "type": "float", "description": "积分下限"},
        {"name": "b", "type": "float", "description": "积分上限"},
        {"name": "n", "type": "int", "description": "分段数",
         "required": False, "default": 1000},
    ],
)
def integrate_num(expression: str, a: float, b: float,
                  n: int = 1000) -> Dict[str, Any]:
    if n < 1:
        return {"status": "error", "reason": "分段数 n 必须 >= 1"}
    try:
        h = (b - a) / n
        total = 0.5 * (_safe_eval(expression, a) + _safe_eval(expression, b))
        for i in range(1, n):
            total += _safe_eval(expression, a + i * h)
        result = total * h
        return {"status": "success", "integral": result,
                "method": "trapezoidal", "n": n, "interval": [a, b]}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 7. differentiate —— 数值微分（中心差分）
# ===========================================================================
@register_tool(
    "differentiate", "数值微分（中心差分法），表达式以 x 为变量", "math_compute",
    params=[
        {"name": "expression", "type": "str",
         "description": "函数表达式如 sin(x)，变量为 x"},
        {"name": "x", "type": "float", "description": "求导点"},
        {"name": "h", "type": "float", "description": "步长",
         "required": False, "default": 1e-6},
    ],
)
def differentiate(expression: str, x: float, h: float = 1e-6) -> Dict[str, Any]:
    try:
        f_plus = _safe_eval(expression, x + h)
        f_minus = _safe_eval(expression, x - h)
        derivative = (f_plus - f_minus) / (2.0 * h)
        return {"status": "success", "derivative": derivative,
                "x": x, "method": "central_difference", "h": h}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 8. optimize_min —— 一维优化求最小值（黄金分割法）
# ===========================================================================
@register_tool(
    "optimize_min", "一维优化求最小值（黄金分割法），表达式以 x 为变量",
    "math_compute",
    params=[
        {"name": "expression", "type": "str",
         "description": "目标函数表达式如 (x-3)**2+1，变量为 x"},
        {"name": "a", "type": "float", "description": "搜索区间左端点"},
        {"name": "b", "type": "float", "description": "搜索区间右端点"},
        {"name": "tol", "type": "float", "description": "容差",
         "required": False, "default": 1e-8},
        {"name": "max_iter", "type": "int", "description": "最大迭代次数",
         "required": False, "default": 500},
    ],
)
def optimize_min(expression: str, a: float, b: float,
                 tol: float = 1e-8, max_iter: int = 500) -> Dict[str, Any]:
    try:
        gr = (math.sqrt(5.0) - 1.0) / 2.0  # 黄金比例 ≈ 0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _safe_eval(expression, c)
        fd = _safe_eval(expression, d)
        for i in range(max_iter):
            if abs(b - a) < tol:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = _safe_eval(expression, c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = _safe_eval(expression, d)
        x_min = (a + b) / 2.0
        f_min = _safe_eval(expression, x_min)
        return {"status": "success", "x_min": x_min, "f_min": f_min,
                "iterations": i + 1, "interval": [a, b]}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ===========================================================================
# 9. fft_simple —— 简单离散傅里叶变换（纯 Python DFT）
# ===========================================================================
@register_tool(
    "fft_simple", "简单离散傅里叶变换 DFT（纯 Python，不依赖 scipy）",
    "math_compute",
    params=[
        {"name": "data", "type": "list",
         "description": "输入实数序列如 [1,1,1,1]"},
        {"name": "inverse", "type": "bool", "description": "是否逆变换(IDFT)",
         "required": False, "default": False},
    ],
)
def fft_simple(data: List[float], inverse: bool = False) -> Dict[str, Any]:
    n = len(data)
    if n == 0:
        return {"status": "error", "reason": "输入数据不能为空"}
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / n if inverse else 1.0
    result_real = [0.0] * n
    result_imag = [0.0] * n
    for k in range(n):
        s_real = 0.0
        s_imag = 0.0
        for t in range(n):
            angle = sign * 2.0 * math.pi * k * t / n
            s_real += data[t] * math.cos(angle)
            s_imag += data[t] * math.sin(angle)
        result_real[k] = s_real * scale
        result_imag[k] = s_imag * scale
    magnitude = [math.sqrt(result_real[k] ** 2 + result_imag[k] ** 2)
                 for k in range(n)]
    return {"status": "success", "real": result_real, "imag": result_imag,
            "magnitude": magnitude, "n": n,
            "transform": "IDFT" if inverse else "DFT"}


# ===========================================================================
# 10. interpolate —— 线性插值
# ===========================================================================
@register_tool(
    "interpolate", "线性插值（给定数据点求任意 x 处的 y）", "math_compute",
    params=[
        {"name": "x_data", "type": "list", "description": "已知 x 坐标列表"},
        {"name": "y_data", "type": "list", "description": "已知 y 坐标列表"},
        {"name": "x_query", "type": "float", "description": "待求插值点的 x"},
    ],
)
def interpolate(x_data: List[float], y_data: List[float],
                x_query: float) -> Dict[str, Any]:
    if len(x_data) != len(y_data) or len(x_data) < 2:
        return {"status": "error", "reason": "x_data 和 y_data 长度需相同且 >= 2"}
    # 按 x 排序
    pairs = sorted(zip(x_data, y_data))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    # 外推或内插
    if x_query <= xs[0]:
        # 左外推
        if len(xs) >= 2:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            y_query = ys[0] + slope * (x_query - xs[0])
        else:
            y_query = ys[0]
        return {"status": "success", "y": y_query, "x": x_query,
                "region": "extrapolation_left"}
    if x_query >= xs[-1]:
        if len(xs) >= 2:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            y_query = ys[-1] + slope * (x_query - xs[-1])
        else:
            y_query = ys[-1]
        return {"status": "success", "y": y_query, "x": x_query,
                "region": "extrapolation_right"}
    # 内插
    for i in range(len(xs) - 1):
        if xs[i] <= x_query <= xs[i + 1]:
            t = (x_query - xs[i]) / (xs[i + 1] - xs[i])
            y_query = ys[i] + t * (ys[i + 1] - ys[i])
            return {"status": "success", "y": y_query, "x": x_query,
                    "region": "interpolation", "bracket": [xs[i], xs[i + 1]]}
    return {"status": "error", "reason": "插值失败"}


# ===========================================================================
# 11. curve_fit_least —— 最小二乘法多项式拟合（纯 Python）
# ===========================================================================
@register_tool(
    "curve_fit_least", "最小二乘法多项式拟合（纯 Python 求解正规方程）",
    "math_compute",
    params=[
        {"name": "x_data", "type": "list", "description": "x 坐标列表"},
        {"name": "y_data", "type": "list", "description": "y 坐标列表"},
        {"name": "degree", "type": "int", "description": "多项式阶数",
         "required": False, "default": 2},
    ],
)
def curve_fit_least(x_data: List[float], y_data: List[float],
                    degree: int = 2) -> Dict[str, Any]:
    n = len(x_data)
    if n != len(y_data) or n < degree + 1:
        return {"status": "error",
                "reason": "数据点数须 >= degree+1 且 x/y 等长"}
    if degree < 0:
        return {"status": "error", "reason": "阶数必须 >= 0"}
    m = degree + 1  # 系数个数
    # 构造正规方程 X^T X c = X^T y
    # X^T X 的 (i,j) 元素 = sum(x^(i+j))
    sums = [0.0] * (2 * degree + 1)
    for k in range(2 * degree + 1):
        sums[k] = sum(xi ** k for xi in x_data)
    rhs = [sum(y_data[j] * x_data[j] ** i for j in range(n))
           for i in range(m)]
    # 构造矩阵
    mat = [[sums[i + j] for j in range(m)] for i in range(m)]
    # 用 linear_solve 的核心逻辑求解
    aug = [list(mat[i]) + [rhs[i]] for i in range(m)]
    for col in range(m):
        pivot_row = max(range(col, m), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            return {"status": "error", "reason": "正规方程奇异，尝试降低阶数"}
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(m):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(m + 1)]
    coeffs = [aug[i][m] for i in range(m)]
    # 计算 R^2
    y_mean = sum(y_data) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_data)
    ss_res = sum(
        (y_data[j] - sum(coeffs[i] * x_data[j] ** i for i in range(m))) ** 2
        for j in range(n)
    )
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"status": "success", "coefficients": coeffs,
            "degree": degree, "r_squared": r_squared}


# ===========================================================================
# 12. statistics_basic —— 基本统计量
# ===========================================================================
@register_tool(
    "statistics_basic", "基本统计量（mean/std/var/min/max/median/sum）",
    "math_compute",
    params=[
        {"name": "data", "type": "list", "description": "数值列表"},
    ],
)
def statistics_basic(data: List[float]) -> Dict[str, Any]:
    if not data:
        return {"status": "error", "reason": "数据不能为空"}
    n = len(data)
    s = sum(data)
    mean = s / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in data) / (n - 1)  # 样本方差
    else:
        variance = 0.0
    std = math.sqrt(variance)
    sorted_data = sorted(data)
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
    return {
        "status": "success",
        "count": n,
        "sum": s,
        "mean": mean,
        "median": median,
        "variance": variance,
        "std": std,
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
    }


# ===========================================================================
# 13. random_sample —— 随机采样（均匀/正态）
# ===========================================================================
@register_tool(
    "random_sample", "随机采样（均匀分布或正态分布）", "math_compute",
    params=[
        {"name": "distribution", "type": "str",
         "description": "分布类型: uniform 或 normal",
         "required": False, "default": "uniform"},
        {"name": "n", "type": "int", "description": "采样数量",
         "required": False, "default": 10},
        {"name": "param1", "type": "float",
         "description": "uniform: 下界; normal: 均值 mu",
         "required": False, "default": 0.0},
        {"name": "param2", "type": "float",
         "description": "uniform: 上界; normal: 标准差 sigma",
         "required": False, "default": 1.0},
        {"name": "seed", "type": "int", "description": "随机种子，-1 表示不设",
         "required": False, "default": -1},
    ],
)
def random_sample(distribution: str = "uniform", n: int = 10,
                  param1: float = 0.0, param2: float = 1.0,
                  seed: int = -1) -> Dict[str, Any]:
    if n < 0:
        return {"status": "error", "reason": "采样数 n 不能为负"}
    if seed >= 0:
        random.seed(seed)
    dist = distribution.lower()
    if dist == "uniform":
        if param2 <= param1:
            return {"status": "error", "reason": "均匀分布上界须大于下界"}
        samples = [random.uniform(param1, param2) for _ in range(n)]
    elif dist in ("normal", "gaussian"):
        if param2 < 0:
            return {"status": "error", "reason": "标准差不能为负"}
        samples = [random.gauss(param1, param2) for _ in range(n)]
    else:
        return {"status": "error",
                "reason": f"不支持的分布: {distribution}，可选 uniform/normal"}
    return {"status": "success", "samples": samples, "n": n,
            "distribution": dist, "params": [param1, param2]}


# ===========================================================================
# 14. histogram_compute —— 计算直方图
# ===========================================================================
@register_tool(
    "histogram_compute", "计算直方图（返回各 bin 的频数与边界）", "math_compute",
    params=[
        {"name": "data", "type": "list", "description": "数值列表"},
        {"name": "bins", "type": "int", "description": "分箱数",
         "required": False, "default": 10},
        {"name": "bin_range", "type": "list",
         "description": "自定义范围 [min,max]，留空则自动",
         "required": False, "default": None},
    ],
)
def histogram_compute(data: List[float], bins: int = 10,
                      bin_range: List[float] = None) -> Dict[str, Any]:
    if not data:
        return {"status": "error", "reason": "数据不能为空"}
    if bins < 1:
        return {"status": "error", "reason": "分箱数 bins 必须 >= 1"}
    if bin_range and len(bin_range) == 2:
        lo, hi = bin_range[0], bin_range[1]
    else:
        lo, hi = min(data), max(data)
    if hi == lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in data:
        if v < lo or v > hi:
            continue
        idx = int((v - lo) / width)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    total = sum(counts)
    density = [c / total / width if total > 0 else 0.0 for c in counts]
    return {"status": "success", "counts": counts, "edges": edges,
            "density": density, "bins": bins, "range": [lo, hi],
            "total": total}


# ===========================================================================
# 15. eigenvalues_2x2 —— 2x2 矩阵特征值（解析解）
# ===========================================================================
@register_tool(
    "eigenvalues_2x2", "2×2 矩阵特征值（解析解，支持实/复特征值）",
    "math_compute",
    params=[
        {"name": "matrix", "type": "list",
         "description": "2×2 矩阵如 [[a,b],[c,d]]"},
    ],
)
def eigenvalues_2x2(matrix: List[List[float]]) -> Dict[str, Any]:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        return {"status": "error", "reason": "输入必须是 2×2 矩阵"}
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]
    # 特征方程: λ² - (a+d)λ + (ad-bc) = 0
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4.0 * det
    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2.0
        lambda2 = (trace - sqrt_disc) / 2.0
        return {"status": "success", "eigenvalues": [lambda1, lambda2],
                "type": "real", "trace": trace, "determinant": det}
    else:
        sqrt_disc = math.sqrt(-discriminant)
        real_part = trace / 2.0
        imag_part = sqrt_disc / 2.0
        return {"status": "success",
                "eigenvalues": [
                    {"real": real_part, "imag": imag_part},
                    {"real": real_part, "imag": -imag_part},
                ],
                "type": "complex", "trace": trace, "determinant": det}


# ---------------------------------------------------------------------------
# 模块自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from aerospace_agent.research_tools.base import get_registry

    reg = get_registry()
    print(f"math_compute 模块已注册工具数: "
          f"{len(reg.list_by_category('math_compute'))}")
    print("工具列表:")
    for name in reg.list_by_category("math_compute"):
        print(f"  - {name}")

    # 快速冒烟测试
    print("\n--- 冒烟测试 ---")
    print(basic_arithmetic("1+2*3"))
    print(matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
    print(matrix_invert([[4, 7], [2, 6]]))
    print(linear_solve([[2, 1], [1, 3]], [3, 5]))
    print(root_find("x**3-x-2", 1, 2))
    print(integrate_num("sin(x)", 0, math.pi, 100))
    print(differentiate("x**2", 3.0))
    print(optimize_min("(x-3)**2+1", 0, 5))
    print(fft_simple([1, 1, 1, 1]))
    print(interpolate([0, 1, 2], [0, 1, 4], 1.5))
    print(curve_fit_least([0, 1, 2, 3], [1, 3, 7, 13], degree=2))
    print(statistics_basic([1, 2, 3, 4, 5]))
    print(random_sample("normal", 5, 0, 1, seed=42))
    print(histogram_compute([1, 2, 3, 4, 5, 5, 5], bins=3))
    print(eigenvalues_2x2([[2, 1], [1, 2]]))
