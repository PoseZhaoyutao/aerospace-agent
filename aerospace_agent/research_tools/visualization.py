"""可视化工具集——10 个基于 matplotlib 的科研绘图工具。

第一性原理：
  1. 每个绘图工具接收数据参数，生成图片保存到指定路径，返回路径
  2. matplotlib 不可用时 graceful fallback，返回文本描述而非崩溃
  3. 统一使用 Agg 后端（非交互式），适合服务器/批处理环境
  4. 所有图形参数均有合理默认值，零配置即可出图
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from aerospace_agent.research_tools.base import register_tool

# ---------------------------------------------------------------------------
# matplotlib 可用性检测（懒加载，import 时不崩溃）
# ---------------------------------------------------------------------------
_MPL_AVAILABLE = False
plt = None
_mpl_figure = None
_mpl_cm = None

try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，适合无显示器环境
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure as _mpl_figure
    import matplotlib.cm as _mpl_cm
    _MPL_AVAILABLE = True
except ImportError:
    pass


def _check_mpl(tool_name: str) -> Optional[Dict[str, Any]]:
    """检查 matplotlib 是否可用，不可用时返回 warning dict。"""
    if not _MPL_AVAILABLE:
        return {
            "status": "warning",
            "reason": "matplotlib不可用",
            "text_fallback": f"工具 '{tool_name}' 需要 matplotlib 才能生成图形。"
                             f"请执行 pip install matplotlib 后重试。",
        }
    return None


def _ensure_dir(path: str) -> bool:
    """确保目标目录存在。"""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    return True


def _finalize_fig(fig, save_path: str, title: str = "") -> Dict[str, Any]:
    """收尾：设置标题、保存、关闭。"""
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"status": "success", "path": save_path,
            "title": title, "backend": "matplotlib"}


# ===========================================================================
# 16. plot_line —— 折线图
# ===========================================================================
@register_tool(
    "plot_line", "折线图（支持多条曲线）", "visualization",
    params=[
        {"name": "x_data", "type": "list", "description": "x 坐标列表"},
        {"name": "y_data", "type": "list",
         "description": "y 坐标列表，或多个系列的列表的列表"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Line Plot"},
        {"name": "xlabel", "type": "str", "description": "x 轴标签",
         "required": False, "default": "x"},
        {"name": "ylabel", "type": "str", "description": "y 轴标签",
         "required": False, "default": "y"},
        {"name": "labels", "type": "list", "description": "各系列图例标签",
         "required": False, "default": None},
        {"name": "save_path", "type": "str",
         "description": "图片保存路径如 output/line.png",
         "required": False, "default": "line_plot.png"},
    ],
)
def plot_line(x_data: List[float], y_data: List,
              title: str = "Line Plot", xlabel: str = "x",
              ylabel: str = "y", labels: List[str] = None,
              save_path: str = "line_plot.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_line")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    # 判断是否多系列
    if y_data and isinstance(y_data[0], list):
        for i, ys in enumerate(y_data):
            label = labels[i] if labels and i < len(labels) else f"series_{i}"
            ax.plot(x_data, ys, marker="o", markersize=3, label=label)
        ax.legend()
    else:
        ax.plot(x_data, y_data, marker="o", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 17. plot_scatter —— 散点图
# ===========================================================================
@register_tool(
    "plot_scatter", "散点图（支持颜色映射）", "visualization",
    params=[
        {"name": "x_data", "type": "list", "description": "x 坐标列表"},
        {"name": "y_data", "type": "list", "description": "y 坐标列表"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Scatter Plot"},
        {"name": "xlabel", "type": "str", "description": "x 轴标签",
         "required": False, "default": "x"},
        {"name": "ylabel", "type": "str", "description": "y 轴标签",
         "required": False, "default": "y"},
        {"name": "colors", "type": "list", "description": "每个点的颜色值列表",
         "required": False, "default": None},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "scatter_plot.png"},
    ],
)
def plot_scatter(x_data: List[float], y_data: List[float],
                 title: str = "Scatter Plot", xlabel: str = "x",
                 ylabel: str = "y", colors: List[float] = None,
                 save_path: str = "scatter_plot.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_scatter")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    if colors:
        sc = ax.scatter(x_data, y_data, c=colors, cmap="viridis", alpha=0.7)
        fig.colorbar(sc, ax=ax, label="value")
    else:
        ax.scatter(x_data, y_data, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 18. plot_bar —— 柱状图
# ===========================================================================
@register_tool(
    "plot_bar", "柱状图（支持分组与堆叠）", "visualization",
    params=[
        {"name": "labels", "type": "list", "description": "类别标签列表"},
        {"name": "values", "type": "list",
         "description": "数值列表，或多组数据的列表的列表"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Bar Chart"},
        {"name": "xlabel", "type": "str", "description": "x 轴标签",
         "required": False, "default": "category"},
        {"name": "ylabel", "type": "str", "description": "y 轴标签",
         "required": False, "default": "value"},
        {"name": "stacked", "type": "bool", "description": "是否堆叠",
         "required": False, "default": False},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "bar_chart.png"},
    ],
)
def plot_bar(labels: List[str], values: List,
             title: str = "Bar Chart", xlabel: str = "category",
             ylabel: str = "value", stacked: bool = False,
             save_path: str = "bar_chart.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_bar")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    import numpy as _np
    x_pos = _np.arange(len(labels))
    if values and isinstance(values[0], list):
        # 多组数据
        width = 0.8 / len(values)
        for i, vals in enumerate(values):
            offset = (i - len(values) / 2 + 0.5) * width
            if stacked:
                if i == 0:
                    ax.bar(x_pos, vals, width, label=f"group_{i}")
                    bottoms = _np.array(vals, dtype=float)
                else:
                    ax.bar(x_pos, vals, width, bottom=bottoms,
                           label=f"group_{i}")
                    bottoms += _np.array(vals, dtype=float)
            else:
                ax.bar(x_pos + offset, vals, width, label=f"group_{i}")
        ax.legend()
    else:
        ax.bar(x_pos, values, 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 19. plot_heatmap —— 热力图
# ===========================================================================
@register_tool(
    "plot_heatmap", "热力图（矩阵可视化）", "visualization",
    params=[
        {"name": "matrix", "type": "list", "description": "二维矩阵数据"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Heatmap"},
        {"name": "xlabel", "type": "str", "description": "x 轴标签",
         "required": False, "default": "column"},
        {"name": "ylabel", "type": "str", "description": "y 轴标签",
         "required": False, "default": "row"},
        {"name": "cmap", "type": "str", "description": "色彩映射如 viridis/coolwarm",
         "required": False, "default": "viridis"},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "heatmap.png"},
    ],
)
def plot_heatmap(matrix: List[List[float]], title: str = "Heatmap",
                 xlabel: str = "column", ylabel: str = "row",
                 cmap: str = "viridis",
                 save_path: str = "heatmap.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_heatmap")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    import numpy as _np
    data = _np.array(matrix)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label="value")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # 标注数值（小矩阵时）
    if data.size <= 100:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if data[i, j] > data.mean() else "black",
                        fontsize=7)
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 20. plot_3d —— 3D 散点/曲面图
# ===========================================================================
@register_tool(
    "plot_3d", "3D 散点图或曲面图", "visualization",
    params=[
        {"name": "x_data", "type": "list", "description": "x 坐标列表"},
        {"name": "y_data", "type": "list", "description": "y 坐标列表"},
        {"name": "z_data", "type": "list", "description": "z 坐标列表"},
        {"name": "plot_type", "type": "str",
         "description": "图表类型: scatter 或 surface",
         "required": False, "default": "scatter"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "3D Plot"},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "plot_3d.png"},
    ],
)
def plot_3d(x_data: List[float], y_data: List[float],
            z_data: List[float], plot_type: str = "scatter",
            title: str = "3D Plot",
            save_path: str = "plot_3d.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_3d")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as _np
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if plot_type == "surface":
        # 需要网格化数据
        xs = _np.array(x_data)
        ys = _np.array(y_data)
        zs = _np.array(z_data)
        if xs.ndim == 1:
            # 尝试根据唯一值构建网格
            x_unique = _np.unique(xs)
            y_unique = _np.unique(ys)
            if len(x_unique) * len(y_unique) == len(zs):
                X, Y = _np.meshgrid(x_unique, y_unique)
                Z = zs.reshape(len(y_unique), len(x_unique))
            else:
                return {"status": "error",
                        "reason": "surface 模式需要网格化数据（x*y==len(z)）"}
        else:
            X, Y, Z = xs, ys, zs
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    else:
        ax.scatter(x_data, y_data, z_data, c=z_data, cmap="viridis",
                   alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 21. plot_contour —— 等高线图
# ===========================================================================
@register_tool(
    "plot_contour", "等高线图（需要网格化数据）", "visualization",
    params=[
        {"name": "x_data", "type": "list", "description": "x 网格坐标（一维或二维）"},
        {"name": "y_data", "type": "list", "description": "y 网格坐标（一维或二维）"},
        {"name": "z_data", "type": "list", "description": "z 值矩阵（二维）"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Contour Plot"},
        {"name": "levels", "type": "int", "description": "等高线层数",
         "required": False, "default": 20},
        {"name": "filled", "type": "bool", "description": "是否填充颜色",
         "required": False, "default": True},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "contour.png"},
    ],
)
def plot_contour(x_data: List, y_data: List, z_data: List[List[float]],
                 title: str = "Contour Plot", levels: int = 20,
                 filled: bool = True,
                 save_path: str = "contour.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_contour")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    import numpy as _np
    fig, ax = plt.subplots(figsize=(8, 6))
    x_arr = _np.array(x_data)
    y_arr = _np.array(y_data)
    z_arr = _np.array(z_data)
    # 如果 x/y 是一维，构建网格
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        X, Y = _np.meshgrid(x_arr, y_arr)
    else:
        X, Y = x_arr, y_arr
    if filled:
        cp = ax.contourf(X, Y, z_arr, levels=levels, cmap="RdYlBu_r")
    else:
        cp = ax.contour(X, Y, z_arr, levels=levels, cmap="RdYlBu_r")
    fig.colorbar(cp, ax=ax, label="value")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 22. plot_histogram —— 直方图
# ===========================================================================
@register_tool(
    "plot_histogram", "直方图（频数分布）", "visualization",
    params=[
        {"name": "data", "type": "list", "description": "数据列表"},
        {"name": "bins", "type": "int", "description": "分箱数",
         "required": False, "default": 30},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Histogram"},
        {"name": "xlabel", "type": "str", "description": "x 轴标签",
         "required": False, "default": "value"},
        {"name": "ylabel", "type": "str", "description": "y 轴标签",
         "required": False, "default": "frequency"},
        {"name": "density", "type": "bool", "description": "是否归一化为概率密度",
         "required": False, "default": False},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "histogram.png"},
    ],
)
def plot_histogram(data: List[float], bins: int = 30,
                   title: str = "Histogram", xlabel: str = "value",
                   ylabel: str = "frequency", density: bool = False,
                   save_path: str = "histogram.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_histogram")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, density=density, alpha=0.7,
            color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    return _finalize_fig(fig, save_path, title)


# ===========================================================================
# 23. plot_multi —— 多子图
# ===========================================================================
@register_tool(
    "plot_multi", "多子图（在一个画布中绘制多个子图）", "visualization",
    params=[
        {"name": "plots", "type": "list",
         "description": "子图配置列表，每项为 {type,x,y,title,...}"},
        {"name": "nrows", "type": "int", "description": "子图行数",
         "required": False, "default": 1},
        {"name": "ncols", "type": "int", "description": "子图列数",
         "required": False, "default": 2},
        {"name": "figsize", "type": "list", "description": "画布大小 [w,h]",
         "required": False, "default": None},
        {"name": "title", "type": "str", "description": "总标题",
         "required": False, "default": "Multi-plot"},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "multi_plot.png"},
    ],
)
def plot_multi(plots: List[Dict], nrows: int = 1, ncols: int = 2,
               figsize: List[float] = None, title: str = "Multi-plot",
               save_path: str = "multi_plot.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_multi")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    if figsize is None:
        figsize = [5 * ncols, 4 * nrows]
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(figsize))
    # 统一为可迭代的 axes
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1 or ncols == 1:
        axes = _flatten_axes(axes)
    for idx, plot_cfg in enumerate(plots):
        if idx >= nrows * ncols:
            break
        ax = _get_axis(axes, idx, nrows, ncols)
        ptype = plot_cfg.get("type", "line")
        x = plot_cfg.get("x", [])
        y = plot_cfg.get("y", [])
        sub_title = plot_cfg.get("title", f"subplot_{idx}")
        if ptype == "line":
            ax.plot(x, y, marker="o", markersize=3)
        elif ptype == "scatter":
            ax.scatter(x, y, alpha=0.7)
        elif ptype == "bar":
            ax.bar(range(len(y)), y)
        elif ptype == "hist":
            ax.hist(y, bins=plot_cfg.get("bins", 20), alpha=0.7)
        ax.set_title(sub_title)
        ax.grid(True, alpha=0.3)
    # 隐藏多余子图
    for idx in range(len(plots), nrows * ncols):
        ax = _get_axis(axes, idx, nrows, ncols)
        ax.set_visible(False)
    return _finalize_fig(fig, save_path, title)


def _flatten_axes(axes):
    """将 axes 展平为一维列表的列表。"""
    try:
        flat = list(axes.flat) if hasattr(axes, "flat") else list(axes)
    except Exception:
        flat = [axes]
    return [flat]


def _get_axis(axes, idx, nrows, ncols):
    """根据索引获取对应的 axes 对象。"""
    if nrows == 1 and ncols == 1:
        return axes[0][0] if isinstance(axes[0], list) else axes[0]
    if nrows == 1 or ncols == 1:
        flat = axes[0] if isinstance(axes[0], list) else axes
        return flat[idx]
    r, c = divmod(idx, ncols)
    return axes[r][c]


# ===========================================================================
# 24. save_figure —— 保存图形到文件（通用绘图保存工具）
# ===========================================================================
@register_tool(
    "save_figure", "通用图形保存工具（按描述生成并保存图形到文件）",
    "visualization",
    params=[
        {"name": "plot_config", "type": "dict",
         "description": "绘图配置: {type:line/scatter/bar/hist, x, y, title, ...}"},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "figure.png"},
        {"name": "figsize", "type": "list", "description": "画布大小 [w,h]",
         "required": False, "default": None},
        {"name": "dpi", "type": "int", "description": "分辨率",
         "required": False, "default": 150},
    ],
)
def save_figure(plot_config: Dict, save_path: str = "figure.png",
                figsize: List[float] = None,
                dpi: int = 150) -> Dict[str, Any]:
    fallback = _check_mpl("save_figure")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    if figsize is None:
        figsize = [8, 5]
    fig, ax = plt.subplots(figsize=tuple(figsize))
    ptype = plot_config.get("type", "line")
    x = plot_config.get("x", [])
    y = plot_config.get("y", [])
    title = plot_config.get("title", "Figure")
    xlabel = plot_config.get("xlabel", "x")
    ylabel = plot_config.get("ylabel", "y")
    if ptype == "line":
        ax.plot(x, y, marker="o", markersize=3)
    elif ptype == "scatter":
        ax.scatter(x, y, alpha=0.7)
    elif ptype == "bar":
        labels = plot_config.get("labels", range(len(y)))
        ax.bar(labels, y)
    elif ptype == "hist":
        bins = plot_config.get("bins", 30)
        ax.hist(y if y else x, bins=bins, alpha=0.7)
    elif ptype == "fill":
        ax.fill_between(x, y, alpha=0.5)
    else:
        ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"status": "success", "path": save_path, "type": ptype,
            "title": title, "dpi": dpi, "backend": "matplotlib"}


# ===========================================================================
# 25. plot_surface —— 曲面图
# ===========================================================================
@register_tool(
    "plot_surface", "3D 曲面图（需要网格化 x/y/z 数据）", "visualization",
    params=[
        {"name": "x_data", "type": "list",
         "description": "x 网格坐标（一维唯一值或二维网格）"},
        {"name": "y_data", "type": "list",
         "description": "y 网格坐标（一维唯一值或二维网格）"},
        {"name": "z_data", "type": "list", "description": "z 值矩阵（二维）"},
        {"name": "title", "type": "str", "description": "图表标题",
         "required": False, "default": "Surface Plot"},
        {"name": "cmap", "type": "str", "description": "色彩映射",
         "required": False, "default": "plasma"},
        {"name": "save_path", "type": "str", "description": "图片保存路径",
         "required": False, "default": "surface.png"},
    ],
)
def plot_surface(x_data: List, y_data: List, z_data: List[List[float]],
                 title: str = "Surface Plot", cmap: str = "plasma",
                 save_path: str = "surface.png") -> Dict[str, Any]:
    fallback = _check_mpl("plot_surface")
    if fallback:
        return fallback
    _ensure_dir(save_path)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as _np
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    x_arr = _np.array(x_data)
    y_arr = _np.array(y_data)
    z_arr = _np.array(z_data)
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        X, Y = _np.meshgrid(x_arr, y_arr)
    else:
        X, Y = x_arr, y_arr
    surf = ax.plot_surface(X, Y, z_arr, cmap=cmap, alpha=0.85,
                           edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return _finalize_fig(fig, save_path, title)


# ---------------------------------------------------------------------------
# 模块自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from aerospace_agent.research_tools.base import get_registry

    reg = get_registry()
    print(f"visualization 模块已注册工具数: "
          f"{len(reg.list_by_category('visualization'))}")
    print("工具列表:")
    for name in reg.list_by_category("visualization"):
        print(f"  - {name}")

    if _MPL_AVAILABLE:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        print("\n--- 冒烟测试 ---")
        print(plot_line([1, 2, 3, 4], [1, 4, 9, 16],
                        save_path=os.path.join(tmpdir, "line.png")))
        print(plot_scatter([1, 2, 3], [4, 5, 6], colors=[1, 2, 3],
                           save_path=os.path.join(tmpdir, "scatter.png")))
        print(plot_bar(["A", "B", "C"], [3, 7, 5],
                       save_path=os.path.join(tmpdir, "bar.png")))
        print(plot_heatmap([[1, 2], [3, 4]],
                           save_path=os.path.join(tmpdir, "heat.png")))
        print(plot_3d([1, 2, 3], [4, 5, 6], [7, 8, 9],
                      save_path=os.path.join(tmpdir, "3d.png")))
        print(plot_contour([0, 1, 2], [0, 1, 2],
                           [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                           save_path=os.path.join(tmpdir, "contour.png")))
        print(plot_histogram([1, 2, 2, 3, 3, 3, 4, 4, 5],
                             save_path=os.path.join(tmpdir, "hist.png")))
        print(plot_multi(
            [{"type": "line", "x": [1, 2, 3], "y": [1, 4, 9], "title": "y=x^2"},
             {"type": "scatter", "x": [1, 2, 3], "y": [3, 2, 1], "title": "dec"}],
            save_path=os.path.join(tmpdir, "multi.png")))
        print(save_figure({"type": "bar", "x": [0, 1, 2], "y": [5, 3, 7],
                           "title": "demo"},
                          save_path=os.path.join(tmpdir, "saved.png")))
        print(plot_surface([0, 1, 2], [0, 1, 2],
                           [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                           save_path=os.path.join(tmpdir, "surf.png")))
    else:
        print("\nmatplotlib 不可用，仅测试 fallback:")
        print(plot_line([1, 2], [3, 4]))
