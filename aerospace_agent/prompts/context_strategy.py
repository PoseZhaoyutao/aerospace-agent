"""CEO 上下文管理策略（代码化）。

将 ``ContextManager`` 的 Compress / Essential-preserve / Offload 三层策略
编码为可程序化访问的常量与函数，供 Agent 在 ReAct 循环中动态决策。

CEO 三层结构
------------
- Essential 层：任务规格、关键公式、用户原始指令——永不压缩，原样保留
- Compress  层：中间对话历史、工具调用记录——超阈值时摘要压缩
- Offload   层：大块数据、检索结果、轨迹数据——存外部文件，上下文只保留引用

本模块提供：
- ``CONTEXT_STRATEGY``         : 完整策略字典（阈值/触发条件/动作）
- ``ESSENTIAL_PRESERVE_RULES`` : Essential 层永不压缩的内容清单
- ``COMPRESS_TRIGGER``         : 压缩触发条件
- ``OFFLOAD_TRIGGER``          : 卸载触发条件
- ``get_context_prompt()``     : 根据当前 Agent 状态生成上下文感知提示
- ``decide_action()``          : 根据状态决策应执行的动作 (keep/compress/offload)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "CONTEXT_STRATEGY",
    "ESSENTIAL_PRESERVE_RULES",
    "COMPRESS_TRIGGER",
    "OFFLOAD_TRIGGER",
    "get_context_prompt",
    "decide_action",
]


# ---------------------------------------------------------------------------
# Essential 层：永不压缩的内容清单
# ---------------------------------------------------------------------------
ESSENTIAL_PRESERVE_RULES: Dict[str, Any] = {
    "description": "以下内容必须在 Essential 层原样保留，任何情况下不得压缩、截断或摘要",
    "never_compress": [
        "用户原始任务指令（逐字保留）",
        "任务设计约束（精度要求、Δv 预算、转移时间上限等硬性指标）",
        "关键物理常数与公式（mu、R_earth、vis-viva 方程等）",
        "安全边界条件（再入过载上限、碰撞概率阈值、燃料余量下限）",
        "用户指定的参考系与历元（frame、epoch、time_scale）",
        "已确认的最终答案（Final Answer 一旦给出不得在后续压缩中丢失）",
        "Loop 引擎的 workflow_id 与 validation_report 结论",
        "交叉验证的置信度等级与误差阈值",
    ],
    "preserve_rule": "即使 token 预算超限，Essential 层也完整保留；超出部分从 Compress 层截断。",
}


# ---------------------------------------------------------------------------
# Compress 层：压缩触发条件
# ---------------------------------------------------------------------------
COMPRESS_TRIGGER: Dict[str, Any] = {
    "description": "Compress 层（对话历史 + 工具记录）的压缩触发条件",
    "token_threshold": 2000,        # Compress 层 token 超过此值触发压缩
    "round_threshold": 6,           # ReAct 轮次超过此值触发压缩（保留最近 N 条原文）
    "keep_recent": 6,               # 压缩时保留最近 N 条消息/记录原文
    "compress_method": "head摘要 + tail原文",
    "head_summary_len": 60,         # 旧消息摘要保留前 N 字符
    "trigger_condition": (
        "compress_tokens > token_threshold OR "
        "message_count > round_threshold"
    ),
    "action": "对旧消息保留角色标记 + 前 60 字摘要，最近 6 条原文保留",
}


# ---------------------------------------------------------------------------
# Offload 层：卸载触发条件
# ---------------------------------------------------------------------------
OFFLOAD_TRIGGER: Dict[str, Any] = {
    "description": "Offload 层的卸载触发条件——大块数据存外部文件，上下文只留引用",
    "data_size_threshold": 4096,    # 单条数据字符数超过此值触发卸载
    "token_threshold": 1500,        # 单条数据 token 超过此值触发卸载
    "time_series_threshold": 50,    # 时序数据点数超过此值触发卸载（如 state_history）
    "trigger_data_types": [
        "state_history",            # 轨道传播时序状态
        "access_windows",           # 地面可见性窗口列表
        "trajectory_points",        # 轨迹点序列
        "rag_results",              # RAG 检索结果全文
        "literature_papers",        # 文献全文与 PDF
        "porkchop_data",            # porkchop 图数据
        "ephemeris_table",          # 星历表
    ],
    "offload_action": "save_offload(key, data) -> 上下文只保留 key + 摘要 + 路径引用",
    "context_ref_format": "[{key}] -> {path} | 摘要: {summary} | {size} 字节",
}


# ---------------------------------------------------------------------------
# 完整策略字典（汇总三层）
# ---------------------------------------------------------------------------
CONTEXT_STRATEGY: Dict[str, Any] = {
    "name": "CEO",
    "full_name": "Compress / Essential-preserve / Offload",
    "layers": {
        "essential": {
            "policy": "原样保留，永不压缩",
            "rules": ESSENTIAL_PRESERVE_RULES,
        },
        "compress": {
            "policy": "超阈值摘要压缩，保留最近原文",
            "trigger": COMPRESS_TRIGGER,
        },
        "offload": {
            "policy": "大块数据存外部文件，上下文只留引用",
            "trigger": OFFLOAD_TRIGGER,
        },
    },
    "token_budget_default": 8000,   # 默认总 token 预算
    "priority": ["essential", "offload_ref", "compress"],
    "description": (
        "Essential 层永远完整保留 → Offload 层只放引用 → Compress 层先压缩再按"
        "剩余预算从新到旧保留，超出部分截断（仅截断 Compress，绝不截断 Essential）"
    ),
}


# ---------------------------------------------------------------------------
# 决策函数：根据 Agent 状态决定应执行的动作
# ---------------------------------------------------------------------------
def decide_action(agent_state: Dict[str, Any]) -> str:
    """根据当前 Agent 状态决策上下文管理动作。

    Args:
        agent_state: Agent 状态字典，可包含以下键：
            - ``compress_tokens``  : Compress 层当前 token 数
            - ``message_count``    : 当前消息条数
            - ``tool_records``     : 工具调用记录数
            - ``last_data_size``   : 最近一条工具结果字符数
            - ``last_data_type``   : 最近一条工具结果数据类型
            - ``last_data_points`` : 最近一条时序数据点数
            - ``offload_count``    : 已卸载条目数

    Returns:
        动作字符串之一：
        - ``"keep"``     : 无需处理
        - ``"compress"`` : 触发 Compress 层压缩
        - ``"offload"``  : 触发 Offload 层卸载
        - ``"both"``     : 同时压缩与卸载
    """
    compress_tokens = agent_state.get("compress_tokens", 0)
    message_count = agent_state.get("message_count", 0)
    last_data_size = agent_state.get("last_data_size", 0)
    last_data_type = agent_state.get("last_data_type", "")
    last_data_points = agent_state.get("last_data_points", 0)

    need_compress = (
        compress_tokens > COMPRESS_TRIGGER["token_threshold"]
        or message_count > COMPRESS_TRIGGER["round_threshold"]
    )
    need_offload = (
        last_data_size > OFFLOAD_TRIGGER["data_size_threshold"]
        or last_data_type in OFFLOAD_TRIGGER["trigger_data_types"]
        or last_data_points > OFFLOAD_TRIGGER["time_series_threshold"]
    )

    if need_compress and need_offload:
        return "both"
    if need_compress:
        return "compress"
    if need_offload:
        return "offload"
    return "keep"


# ---------------------------------------------------------------------------
# 上下文感知提示生成
# ---------------------------------------------------------------------------
def get_context_prompt(agent_state: Optional[Dict[str, Any]] = None) -> str:
    """根据当前 Agent 状态生成上下文感知提示词。

    该提示词可追加到 ReAct 系统指令末尾，指导 Agent 在当前上下文压力下
    采取正确的压缩/卸载/保留策略。

    Args:
        agent_state: Agent 状态字典（结构同 ``decide_action``）。
            为 None 或空时返回静态策略说明。

    Returns:
        上下文感知提示词字符串。
    """
    base = (
        "# 当前上下文管理状态（CEO 三层）\n"
        "策略：Essential 层原样保留 → Offload 层只留引用 → "
        "Compress 层超阈值压缩。\n"
        f"- Compress 触发阈值：{COMPRESS_TRIGGER['token_threshold']} token "
        f"或 {COMPRESS_TRIGGER['round_threshold']} 轮\n"
        f"- Offload 触发阈值：单条数据 > {OFFLOAD_TRIGGER['data_size_threshold']} 字符 "
        f"或时序点 > {OFFLOAD_TRIGGER['time_series_threshold']}\n"
        "- Essential 层永不压缩：任务规格、关键公式、用户指令、安全边界、最终答案。"
    )

    if not agent_state:
        return base + "\n\n（未提供实时状态，按静态策略执行。）"

    action = decide_action(agent_state)
    compress_tokens = agent_state.get("compress_tokens", 0)
    message_count = agent_state.get("message_count", 0)
    tool_records = agent_state.get("tool_records", 0)
    offload_count = agent_state.get("offload_count", 0)
    last_data_type = agent_state.get("last_data_type", "")
    last_data_size = agent_state.get("last_data_size", 0)

    status_lines = [
        f"- Compress 层：{compress_tokens} token, {message_count} 条消息, "
        f"{tool_records} 条工具记录",
        f"- Offload 层：已卸载 {offload_count} 条",
    ]
    if last_data_type:
        status_lines.append(
            f"- 最近工具结果：类型={last_data_type}, 大小={last_data_size} 字符"
        )

    advice = ""
    if action == "compress":
        advice = (
            "\n\n【上下文压力提示】Compress 层已超阈值。请在下一轮 Thought 中"
            "确认：旧的工具结果已不再需要细节，可接受摘要压缩。"
            "Essential 层（任务规格/公式/最终答案）不受影响。"
        )
    elif action == "offload":
        advice = (
            "\n\n【卸载提示】最近一条工具结果数据量较大。请将该数据通过 "
            "offload 存入外部文件，上下文只保留引用 (key + 摘要 + 路径)。"
            "后续如需细节可按 key 重新加载。"
        )
    elif action == "both":
        advice = (
            "\n\n【上下文压力 + 卸载提示】Compress 层超阈值且最近数据量大。"
            "请先将大块数据 offload 到外部文件，再接受旧消息的摘要压缩。"
            "Essential 层始终原样保留。"
        )
    else:
        advice = (
            "\n\n【上下文正常】当前各层均在阈值内，无需额外压缩或卸载操作。"
        )

    return base + "\n" + "\n".join(status_lines) + advice
