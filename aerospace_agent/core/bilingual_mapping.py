"""双语关键词映射 — 解决向量检索跨语言失败问题。

SimpleEmbedder 用字符 n-gram，中英文零字符重叠。
本模块提供中英双语关键词字典 + 混合检索策略：
    1. 将中文查询翻译为英文关键词
    2. 对工具名称和描述做关键词匹配
    3. 与向量检索结果合并去重
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


# ======================================================================
# 中英双语关键词字典
# ======================================================================

BILINGUAL_MAP: Dict[str, List[str]] = {
    # 数学计算
    "计算": ["calculate", "compute", "math", "calculator", "arithmetic"],
    "运算": ["calculate", "compute", "arithmetic"],
    "求值": ["evaluate", "calculate"],
    "乘方": ["power", "exponent", "pow"],
    "次方": ["power", "exponent", "pow"],
    "开方": ["sqrt", "root", "square"],
    "平方": ["square", "power"],
    "立方": ["cube", "power"],
    "对数": ["log", "logarithm"],
    "指数": ["exponent", "exp"],
    "三角": ["trig", "sin", "cos", "tan"],
    "正弦": ["sin", "sine"],
    "余弦": ["cos", "cosine"],
    "正切": ["tan", "tangent"],
    "矩阵": ["matrix"],
    "行列式": ["determinant"],
    "特征值": ["eigenvalue", "eigen"],
    "逆矩阵": ["invert", "inverse"],
    "转置": ["transpose"],
    "积分": ["integral", "integrate"],
    "微分": ["derivative", "differentiate"],
    "导数": ["derivative"],
    "方程": ["equation", "solve"],
    "求解": ["solve", "solution"],
    "根": ["root", "solve"],
    "插值": ["interpolate", "interpolation"],
    "拟合": ["fit", "curve_fit"],
    "统计": ["statistics", "stat"],
    "均值": ["mean", "average"],
    "方差": ["variance"],
    "标准差": ["std", "standard_deviation"],
    "概率": ["probability"],
    "分布": ["distribution"],
    "随机": ["random"],
    # 文件操作
    "保存": ["save", "write", "store"],
    "写入": ["write", "save"],
    "读取": ["read", "load"],
    "读入": ["read", "load"],
    "打开": ["open", "read"],
    "删除": ["delete", "remove"],
    "复制": ["copy"],
    "移动": ["move", "rename"],
    "重命名": ["rename", "move"],
    "创建目录": ["mkdir", "create_directory", "makedirs"],
    "列出": ["list", "glob", "enumerate"],
    "查找文件": ["search", "find", "glob", "grep"],
    "搜索": ["search", "find", "grep"],
    "文件": ["file"],
    "目录": ["directory", "folder", "path"],
    "路径": ["path"],
    # 数据处理
    "解析": ["parse", "extract"],
    "提取": ["extract", "parse"],
    "过滤": ["filter"],
    "排序": ["sort"],
    "分组": ["group", "groupby"],
    "合并": ["merge", "join", "concat"],
    "拼接": ["concat", "join"],
    "分割": ["split"],
    "转换": ["convert", "transform"],
    "格式化": ["format"],
    "编码": ["encode"],
    "解码": ["decode"],
    "压缩": ["compress", "zip"],
    "解压": ["decompress", "unzip"],
    # 可视化
    "画图": ["plot", "chart", "visualize"],
    "绘图": ["plot", "chart", "draw"],
    "折线图": ["plot_line", "line", "plot"],
    "柱状图": ["bar", "plot_bar"],
    "散点图": ["scatter", "plot_scatter"],
    "饼图": ["pie", "plot_pie"],
    "直方图": ["histogram", "hist"],
    "热力图": ["heatmap"],
    "等高线": ["contour"],
    "三维": ["3d", "surface"],
    "可视化": ["visualize", "plot"],
    # 文本处理
    "文本": ["text"],
    "字符串": ["string", "text"],
    "正则": ["regex", "pattern"],
    "替换": ["replace", "substitute"],
    "匹配": ["match", "find"],
    "分割文本": ["text_split", "split"],
    "连接文本": ["text_join", "concat"],
    "大小写": ["case", "upper", "lower"],
    "去除空白": ["strip", "trim"],
    # 网络操作
    "下载": ["download", "fetch"],
    "上传": ["upload"],
    "请求": ["request", "http"],
    "网页": ["web", "url", "html"],
    "链接": ["url", "link"],
    # 代码执行
    "运行": ["run", "execute", "python"],
    "执行": ["execute", "run"],
    "脚本": ["script", "python"],
    "编译": ["compile"],
    "调试": ["debug"],
    # 科研参考
    "文献": ["literature", "paper", "reference"],
    "论文": ["paper", "literature"],
    "引用": ["citation", "cite"],
    "公式": ["formula"],
    "常数": ["constant"],
    "单位": ["unit", "convert"],
    "物理": ["physics"],
    "化学": ["chemistry"],
    "生物": ["biology"],
    # 航天专用
    "轨道": ["orbit", "orbital"],
    "速度": ["velocity", "speed"],
    "加速度": ["acceleration"],
    "角度": ["angle"],
    "弧度": ["radian", "rad"],
    "度": ["degree", "deg"],
    "坐标": ["coordinate"],
    "位置": ["position"],
    "姿态": ["attitude"],
    "导航": ["navigation"],
    "变轨": ["maneuver", "transfer"],
    "转移": ["transfer"],
    "霍曼": ["hohmann"],
    "月球": ["lunar", "moon"],
    "地球": ["earth", "terrestrial"],
    "发射": ["launch"],
    "再入": ["reentry"],
    "引力": ["gravity", "gravitational"],
    "质量": ["mass"],
    "能量": ["energy"],
    "动量": ["momentum"],
    "力": ["force"],
    "时间": ["time"],
    "日期": ["date"],
    # 系统操作
    "环境变量": ["env", "environment"],
    "进程": ["process"],
    "系统": ["system"],
    "命令": ["command", "shell"],
    "Git": ["git"],
    # 自我进化
    "创建工具": ["create_tool", "create"],
    "列出工具": ["list_tools"],
    "帮助": ["help", "tool_help"],
}


# ======================================================================
# 反向映射（英文→中文，用于扩展搜索）
# ======================================================================

def _build_reverse_map() -> Dict[str, List[str]]:
    """构建英文→中文反向映射。"""
    reverse: Dict[str, List[str]] = {}
    for cn, en_list in BILINGUAL_MAP.items():
        for en in en_list:
            if en not in reverse:
                reverse[en] = []
            reverse[en].append(cn)
    return reverse

REVERSE_MAP = _build_reverse_map()


# ======================================================================
# 查询翻译
# ======================================================================

def translate_query(query: str) -> List[str]:
    """将查询翻译为英文关键词列表。

    策略：
    1. 扫描查询中的中文关键词
    2. 对每个匹配的中文关键词，添加对应的英文翻译
    3. 保留查询中原有的英文词
    """
    keywords: Set[str] = set()
    query_lower = query.lower()

    # 1. 匹配中文关键词
    for cn_word, en_words in BILINGUAL_MAP.items():
        if cn_word in query:
            for en in en_words:
                keywords.add(en)

    # 2. 提取查询中的英文词（>= 3 字符）
    en_words = re.findall(r'[a-zA-Z_]{3,}', query_lower)
    for w in en_words:
        keywords.add(w)

    return list(keywords)


# ======================================================================
# 混合检索 — 关键词匹配 + 向量检索
# ======================================================================

def keyword_match_score(
    query: str,
    tool_name: str,
    tool_description: str = "",
) -> float:
    """计算查询与工具的关键词匹配分数。

    返回 0.0-1.0 的分数。

    评分策略：
    - 工具名精确匹配关键词 → 2.0（强信号）
    - 工具名包含关键词 → 0.8（中信号）
    - 描述包含关键词 → 0.3（弱信号）
    - 归一化：除以 min(翻译词数, 3)，避免词数过多稀释分数
    """
    translated = translate_query(query)
    if not translated:
        return 0.0

    name_lower = tool_name.lower()
    desc_lower = tool_description.lower()

    raw_score = 0.0
    for kw in translated:
        kw_lower = kw.lower()
        # 工具名精确匹配 → 强信号
        if kw_lower == name_lower:
            raw_score += 2.0
        # 工具名包含关键词 → 中信号
        elif kw_lower in name_lower:
            raw_score += 0.8
        # 描述包含关键词 → 弱信号
        elif kw_lower in desc_lower:
            raw_score += 0.3

    # 归一化：除以 min(翻译词数, 3)，避免词数过多稀释
    normalize_by = min(len(translated), 3)
    return min(1.0, raw_score / normalize_by) if normalize_by > 0 else 0.0


def hybrid_search(
    query: str,
    tools: List[Tuple[str, str]],  # [(name, description), ...]
    vector_results: List[Tuple[str, float]],  # [(name, score), ...]
    k: int = 10,
    keyword_weight: float = 0.6,
    vector_weight: float = 0.4,
) -> List[Dict]:
    """混合检索 — 关键词匹配 + 向量检索。

    Args:
        query: 用户查询
        tools: 所有工具的 (name, description) 列表
        vector_results: 向量检索结果 [(name, score), ...]
        k: 返回 top-K
        keyword_weight: 关键词匹配权重
        vector_weight: 向量检索权重

    Returns:
        [{"name": str, "score": float, "source": str}, ...]
    """
    # 1. 向量检索分数
    vector_scores: Dict[str, float] = {}
    for name, score in vector_results:
        vector_scores[name] = score

    # 2. 关键词匹配分数
    keyword_scores: Dict[str, float] = {}
    for name, desc in tools:
        ks = keyword_match_score(query, name, desc)
        if ks > 0:
            keyword_scores[name] = ks

    # 3. 合并分数
    all_names = set(vector_scores.keys()) | set(keyword_scores.keys())
    results: List[Dict] = []
    for name in all_names:
        vs = vector_scores.get(name, 0.0)
        ks = keyword_scores.get(name, 0.0)
        combined = vector_weight * vs + keyword_weight * ks
        source = "both" if vs > 0 and ks > 0 else ("keyword" if ks > 0 else "vector")
        results.append({
            "name": name,
            "score": combined,
            "source": source,
            "vector_score": vs,
            "keyword_score": ks,
        })

    # 4. 排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
