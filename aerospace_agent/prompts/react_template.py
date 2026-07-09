"""ReAct 推理循环提示词模板。

定义 ReAct (Reason+Act) 循环各阶段所需的提示词模板，统一通过
``str.format()`` 注入运行时变量。所有模板遵循航天动力学领域的单位/参考系/
历元声明规范。

模板清单
--------
- ``REACT_SYSTEM``              : ReAct 循环系统指令（Thought/Action/Observation/Final Answer 格式约束）
- ``REACT_USER_TEMPLATE``       : 用户任务输入模板（含可用工具列表与上下文）
- ``REACT_OBSERVATION_TEMPLATE``: 工具结果回填模板（将 Observation 注入下一轮）
- ``FORMAT_GUIDE``             : 正确 ReAct 格式示例（含正反例）

占位符说明
----------
- ``{task}``             : 用户任务描述
- ``{available_tools}``  : 可用工具列表文本
- ``{context_block}``    : CEO 上下文块
- ``{step}``             : 当前 ReAct 步数
- ``{max_steps}``        : 最大步数
- ``{tool_name}``        : 工具名称
- ``{tool_result}``      : 工具返回结果
- ``{round}``            : 当前 Loop 轮次（可选）
"""
from __future__ import annotations

__all__ = [
    "REACT_SYSTEM",
    "REACT_USER_TEMPLATE",
    "REACT_OBSERVATION_TEMPLATE",
    "FORMAT_GUIDE",
    "build_react_messages",
]


# ReAct 循环系统指令：定义推理-行动-观察-回答的闭环格式
REACT_SYSTEM: str = """\
你正在以 ReAct (Reason+Act) 模式执行航天动力学任务。每一轮你必须先「推理」
(Thought)，再决定「行动」(Action + Action Input) 或给出「最终答案」(Final Answer)。

## 严格输出格式（每次仅输出一个 Thought，随后跟 Action 或 Final Answer）

行动格式：
    Thought: <推理过程：分析当前已知信息、为何选择该工具、预期得到什么>
    Action: <工具名，须为可用工具列表中的名称>
    Action Input: <JSON 格式参数，键名须与工具方法签名一致，使用 SI 单位>

终结格式：
    Thought: <推理过程：综合所有 Observation，说明结论如何得出>
    Final Answer: <结构化最终答案，须含 units/frame/time_scale/engine_version>

## 关键约束
1. Action Input 必须是合法 JSON。例如位置/速度数组用 [x, y, z] 表示，
   历元用 {{"value": "...", "scale": "UTC", "format": "ISO"}} 表示。
2. 严禁在同一条回复中同时输出 Action 与 Final Answer。
3. 观察到工具错误 (error 字段非空) 时，须在下一轮 Thought 中分析原因并调整参数重试，
   不可直接作为最终答案。
4. 所有数值参数须标注 SI 单位；角度默认 rad，需用 deg 时须在键名中注明 (如 latitude_deg)。
5. 若任务需要多步且涉及验证/修复，优先评估是否调用 loop_engine.run_loop。
6. 达到最大步数仍未完成时，输出当前最佳结论并标注"未完成"。
"""


# 用户任务输入模板：注入任务、工具列表与上下文
REACT_USER_TEMPLATE: str = """\
【任务】
{task}

【可用工具】
{available_tools}

【当前上下文】
{context_block}

【当前进度】ReAct 第 {step}/{max_steps} 步

请按 ReAct 格式开始：先输出 Thought，再输出 Action 或 Final Answer。"""


# 工具结果回填模板：将 Observation 注入下一轮推理
REACT_OBSERVATION_TEMPLATE: str = """\
Observation: 工具 `{tool_name}` 返回结果如下：
{tool_result}

【进度】ReAct 第 {step}/{max_steps} 步

请基于上述观察继续推理：输出下一轮 Thought，随后跟 Action 或 Final Answer。
注意：若结果含 error 字段，请在 Thought 中分析失败原因并调整参数重试；
若结果已满足任务需求，请给出结构化 Final Answer。"""


# 正确 ReAct 格式示例（含正反例对照）
FORMAT_GUIDE: str = """\
# ReAct 格式正反例

## 正确示例 1：调用轨道传播工具
Thought: 用户需要将 400km 圆轨道传播 1 天。我先调用 propagate_orbit，
使用 GCRF 参考系、二体力学模型，输出步长 300s。初始状态由圆轨道速度公式
v = sqrt(mu/r) 计算，mu 取自 physics.constants (3.986004418e14 m^3/s^2)。
Action: astro_dynamics
Action Input: {{"method": "propagate_orbit", "initial_state_dict": {{"epoch": {{"value": "2026-01-01T00:00:00", "scale": "UTC", "format": "ISO"}}, "frame": {{"name": "GCRF", "center": "Earth"}}, "representation": "cartesian", "position_m": [6778137.0, 0.0, 0.0], "velocity_mps": [0.0, 7668.6, 0.0]}}, "force_model_dict": {{"central_body": "Earth", "gravity": "point_mass"}}, "duration_s": 86400.0, "output_step_s": 300.0, "engine": "orekit"}}

## 正确示例 2：给出最终答案
Thought: 已通过 orekit 传播 86400s，得到 state_history 含 289 个点。
交叉验证位置误差 < 1m，可信度高。综合给出结构化结论。
Final Answer:
## 结果摘要
400km 圆轨道传播 1 天 (86400s) 完成，轨道周期约 92.68 min，完成约 15.5 圈。
## 关键参数
- units: SI (m, m/s, s)
- frame: GCRF (地心天球参考系)
- time_scale: UTC
- engine_version: astro_dynamics_mcp v0.1.0 / orekit
## 误差与可信度
交叉验证 position_error < 1.0 m，confidence=high

## 错误示例 1：同时输出 Action 和 Final Answer（禁止）
Thought: ...
Action: propagate_orbit
Final Answer: ...   <-- 错误：不可与 Action 同时出现

## 错误示例 2：Action Input 非 JSON（禁止）
Action Input: 高度400km 传播1天   <-- 错误：必须是合法 JSON

## 错误示例 3：无参考系/单位（禁止）
Final Answer: 位置 [6778, 0, 0]   <-- 错误：无单位、无参考系、无历元
"""


def build_react_messages(
    task: str,
    available_tools: str,
    context_block: str = "(无额外上下文)",
    step: int = 1,
    max_steps: int = 10,
    system_extra: str = "",
) -> list:
    """构建 ReAct 首轮消息列表 (system + user)。

    Args:
        task: 用户任务描述。
        available_tools: 可用工具列表文本。
        context_block: CEO 上下文块文本。
        step: 当前 ReAct 步数。
        max_steps: 最大步数。
        system_extra: 追加到 system 指令末尾的额外文本（如任务模板）。

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]`` 消息列表。
    """
    sys_text = REACT_SYSTEM
    if system_extra:
        sys_text = sys_text + "\n\n" + system_extra
    user_text = REACT_USER_TEMPLATE.format(
        task=task,
        available_tools=available_tools,
        context_block=context_block,
        step=step,
        max_steps=max_steps,
    )
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
    ]
