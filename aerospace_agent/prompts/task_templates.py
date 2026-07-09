"""任务专属提示词模板。

为七类常见航天动力学任务定义专属提示词模板。每个模板为 ``dict``，结构为：

    {{
        "system":       str,  # 任务专属系统指令（追加到 REACT_SYSTEM 之后）
        "user_template": str,  # 用户任务输入模板（含 {task} 等占位符）
        "tools_hint":    str,  # 推荐使用的工具/技能清单
    }}

模板清单
--------
- ``ORBIT_DESIGN``       : 轨道设计任务
- ``LAUNCH_WINDOW``      : 发射窗口计算
- ``TRAJECTORY_ANALYSIS``: 轨迹分析
- ``GROUND_ACCESS``      : 地面站可见性
- ``LUNAR_TRANSFER``     : 地月转移轨道
- ``CROSS_VALIDATION``   : 交叉验证
- ``LITERATURE_SEARCH``  : 文献检索

所有模板遵循 SI 单位、参考系/历元声明、证据溯源等硬性约束。
``user_template`` 中的占位符通过 ``str.format()`` 注入。
"""
from __future__ import annotations

from typing import Dict

__all__ = [
    "ORBIT_DESIGN",
    "LAUNCH_WINDOW",
    "TRAJECTORY_ANALYSIS",
    "GROUND_ACCESS",
    "LUNAR_TRANSFER",
    "CROSS_VALIDATION",
    "LITERATURE_SEARCH",
    "TASK_TEMPLATES",
    "TASK_TYPES",
]


# ---------------------------------------------------------------------------
# 1. 轨道设计
# ---------------------------------------------------------------------------
ORBIT_DESIGN: Dict[str, str] = {
    "system": """\
【轨道设计任务专属指令】
你的目标是根据任务约束（高度、倾角、偏心率、覆盖需求等）设计满足要求的轨道。
设计流程：
  1. 明确约束：半长轴 a、偏心率 e、倾角 i、RAAN Ω、近地点幅角 ω、平近点角 M。
  2. 用 convert_orbit_representation 在笛卡尔/开普勒之间转换，验证一致性。
  3. 用 propagate_orbit 传播至少一个轨道周期，确认轨道稳定。
  4. 用 query_ephemeris_state 获取参考天体（如 J2000 月球）状态以校准历元。
  5. 输出须含轨道六根数 (a, e, i, Ω, ω, M) + 笛卡尔状态 (r, v) + 周期 + 速度。
  6. 通过 RAG 检索轨道设计准则（如太阳同步轨道的 Ω 漂移率）作为证据支撑。
约束：倾角/RAAN 用 rad，半长轴用 m，速度用 m/s；J2 摄动需在 force_model 中开启。""",
    "user_template": """\
【轨道设计任务】
{task}

输入提示（按需提供，缺失项请先调用工具获取）：
- 目标轨道类型：太阳同步 / 圆轨道 / 椭圆轨道 / 冻结轨道 / 摩西轨道
- 约束参数：高度 (m)、倾角 (rad)、偏心率、覆盖纬度范围
- 历元：ISO 8601 字符串 + 时间尺度 (UTC/TDB)
- 参考系：默认 GCRF，地面相关分析需转 ITRF

期望输出：
- 轨道六根数 (a, e, i, Ω, ω, M) 及 SI 单位
- 笛卡尔状态 (position_m, velocity_mps) + frame + epoch
- 轨道周期 (s) 与圆轨道速度 (m/s)
- 设计依据（RAG 证据引用）""",
    "tools_hint": (
        "推荐工具：astro_dynamics(propagate_orbit, convert_orbit_representation, "
        "query_ephemeris_state, transform_frame) | RAG(query) | "
        "可选：loop_engine.run_loop（多步设计+验证）"
    ),
}


# ---------------------------------------------------------------------------
# 2. 发射窗口计算
# ---------------------------------------------------------------------------
LAUNCH_WINDOW: Dict[str, str] = {
    "system": """\
【发射窗口计算专属指令】
你的目标是计算满足轨道倾角、相位、光照、地面测控等约束的发射窗口。
计算流程：
  1. 用 query_ephemeris_state 查询目标天体（如月球）在候选发射日期的状态。
  2. 用 convert_time 将发射日期在 UTC / TDB 间转换，确保历元一致。
  3. 用 propagate_orbit 模拟从停泊轨道到转移轨道的相位演化。
  4. 综合升交点赤经匹配、beta 角（阳光与轨道面夹角）、地面站可见性筛选窗口。
  5. 输出窗口列表：每个窗口含发射时刻 (UTC)、持续时间、约束满足度评分。
  6. 大时间跨度扫描结果（如 60 天逐点）须 offload 到外部文件，上下文只保留摘要。
约束：发射时刻用 UTC ISO 8601；beta 角用 rad；窗口评分 0~1。""",
    "user_template": """\
【发射窗口计算任务】
{task}

输入提示：
- 目标：月球 / 火星 / 特定轨道交会
- 发射场纬度 (rad) 与经度 (rad)
- 停泊轨道参数：高度 (m)、倾角 (rad)
- 搜索起止日期 (UTC ISO 8601)
- 约束：beta 角范围、地面站可见性、光照条件

期望输出：
- 发射窗口列表（发射时刻 UTC + 持续时间 + 评分 + 约束满足情况）
- 最优窗口标注及选择理由
- beta 角随时间变化曲线（offload 大数据后引用）
- 参考系与时间尺度声明""",
    "tools_hint": (
        "推荐工具：astro_dynamics(query_ephemeris_state, convert_time, "
        "propagate_orbit, compute_ground_access) | CEO Offload(轨迹序列) | "
        "RAG(发射窗口方法学)"
    ),
}


# ---------------------------------------------------------------------------
# 3. 轨迹分析
# ---------------------------------------------------------------------------
TRAJECTORY_ANALYSIS: Dict[str, str] = {
    "system": """\
【轨迹分析专属指令】
你的目标是对给定初始状态进行轨道传播与轨迹特性分析。
分析流程：
  1. 用 propagate_orbit 传播指定时长，输出 state_history（时序状态）。
  2. 用 transform_frame 在 GCRF / ITRF / RSW 等参考系间转换，分析地面轨迹。
  3. 用 convert_orbit_representation 提取开普勒根数随时间变化（如 J2 引起的 Ω/ω 漂移）。
  4. 用 cross_validate_results 对比多引擎结果，报告位置/速度误差与可信度。
  5. state_history 超过阈值时必须 offload，上下文只保留首末点与统计摘要。
  6. 输出须含：轨迹类型、周期、近/远地点高度、地面轨迹重复周期、摄动影响。
约束：传播步长用 s；高度用 m；角度漂移率用 rad/s 或 deg/day（标注）。""",
    "user_template": """\
【轨迹分析任务】
{task}

输入提示：
- 初始状态：position_m [x,y,z] + velocity_mps [vx,vy,vz] + frame + epoch
- 力学模型：central_body、gravity（point_mass / J2 / 高阶）、drag、srp、third_body
- 传播时长 (s) 与输出步长 (s)
- 目标参考系（如分析地面轨迹需 ITRF）

期望输出：
- 轨迹特性：周期、近/远地点高度 (m)、偏心率演化
- 地面轨迹重复周期（若涉及）
- 摄动影响分析（J2 漂移率等）
- state_history offload 引用（首末点 + 统计）
- 交叉验证误差与可信度""",
    "tools_hint": (
        "推荐工具：astro_dynamics(propagate_orbit, transform_frame, "
        "convert_orbit_representation, cross_validate_results) | "
        "CEO Offload(state_history) | 可选：loop_engine"
    ),
}


# ---------------------------------------------------------------------------
# 4. 地面站可见性
# ---------------------------------------------------------------------------
GROUND_ACCESS: Dict[str, str] = {
    "system": """\
【地面站可见性专属指令】
你的目标是计算卫星对地面站的可见性窗口。
计算流程：
  1. 用 compute_ground_access 计算指定时段内的 access_windows。
  2. 用 propagate_orbit 提供轨道状态输入（若未直接给出）。
  3. 用 convert_time 确保起止历元时间尺度一致 (UTC)。
  4. 用 transform_frame 将轨道状态转 ITRF 以验证几何关系。
  5. 输出每个可见窗口：起止时刻 (UTC)、最大仰角 (rad)、持续时长 (s)。
  6. 多站或多日结果须 offload，上下文只保留窗口数量与最优窗口摘要。
约束：仰角用 rad（最小仰角阈值如 5deg 须在键名注明 min_elevation_deg）；地面站经纬度用 deg。""",
    "user_template": """\
【地面站可见性任务】
{task}

输入提示：
- 轨道状态：position_m + velocity_mps + frame + epoch（或轨道根数）
- 地面站：name、latitude_deg、longitude_deg、altitude_m、min_elevation_deg
- 搜索时段：start_epoch + stop_epoch (UTC ISO 8601)
- 多站列表（若需对比）

期望输出：
- 可见窗口列表：起止时刻 (UTC)、最大仰角 (rad)、持续时长 (s)
- 每日可见次数统计
- 最优窗口标注（最长持续 / 最高仰角）
- 参考系声明 (orbit frame + station ITRF)""",
    "tools_hint": (
        "推荐工具：astro_dynamics(compute_ground_access, propagate_orbit, "
        "convert_time, transform_frame) | CEO Offload(多站窗口表)"
    ),
}


# ---------------------------------------------------------------------------
# 5. 地月转移轨道
# ---------------------------------------------------------------------------
LUNAR_TRANSFER: Dict[str, str] = {
    "system": """\
【地月转移轨道专属指令】
你的目标是设计地月转移轨道 (LTO)，包含 TLI 机动、转移轨迹与 LOI 捕获。
设计流程：
  1. 用 query_ephemeris_state 查询月球在发射窗口内的状态 (GCRF)。
  2. 用 convert_orbit_representation 计算停泊轨道速度，确定 TLI 速度增量。
  3. 用 propagate_orbit 传播转移轨迹（须开启 third_body=Moon 摄动）。
  4. 用 cross_validate_results 对比 orekit/poliastro/gmat 结果。
  5. 计算中途修正 (TCM) 与 LOI 速度增量预算。
  6. 因流程复杂（多步+验证+修复），强烈建议调用 loop_engine.run_loop 编排。
  7. 转移轨迹 state_history 须 offload，上下文只保留关键节点 (TLI/TCM/LOI)。
约束：C3 用 km²/s²（同时标注 m²/s²）；Δv 用 m/s；转移时间用 s（可附 day）。""",
    "user_template": """\
【地月转移轨道任务】
{task}

输入提示：
- 停泊轨道：高度 (m)、倾角 (rad)
- 发射窗口或目标到达日期 (UTC ISO 8601)
- 转移类型：霍曼 / 自由返回 / 低能转移
- 约束：最大 Δv、转移时间上限、光照条件

期望输出：
- TLI 速度增量 (m/s) 与方向
- 转移时间 (s / day)
- C3 能量 (m²/s² + km²/s²)
- LOI 速度增量 (m/s)
- TCM 规划（次数与预估 Δv）
- 转移轨迹关键节点 (offload 引用)
- 交叉验证误差与可信度""",
    "tools_hint": (
        "强烈推荐：loop_engine.run_loop（端到端编排）| "
        "astro_dynamics(query_ephemeris_state, propagate_orbit, "
        "convert_orbit_representation, cross_validate_results) | "
        "CEO Offload(转移轨迹) | RAG(地月转移方法学)"
    ),
}


# ---------------------------------------------------------------------------
# 6. 交叉验证
# ---------------------------------------------------------------------------
CROSS_VALIDATION: Dict[str, str] = {
    "system": """\
【交叉验证专属指令】
你的目标是用多个引擎对同一任务求解并对比，评估结果一致性与可信度。
验证流程：
  1. 用 check_engine_availability 确认可用引擎 (orekit/poliastro/gmat/basilisk/spiceypy)。
  2. 对同一 task_spec 分别调用 propagate_orbit（engine 参数分别指定）。
  3. 用 cross_validate_results 汇总多引擎结果，输出位置/速度误差与置信度。
  4. 分析误差来源（力学模型差异、积分器差异、常数差异）。
  5. 误差超阈值时，在 Thought 中分析原因并建议改进（如统一常数、提高阶数）。
  6. 输出须含：各引擎结果摘要、误差矩阵、置信度等级 (high/medium/low)、改进建议。
约束：误差用 m 与 m/s；置信度 high=误差<1m、medium=<1km、low=>1km。""",
    "user_template": """\
【交叉验证任务】
{task}

输入提示：
- 待验证任务规格 (task_spec)：含 initial_state、force_model、duration
- 参与引擎列表（默认全部可用引擎）
- 误差阈值 (m)

期望输出：
- 各引擎结果摘要（首末状态 + 关键参数）
- 位置误差 (m) 与速度误差 (m/s) 矩阵
- 置信度等级 (high/medium/low) 与判定依据
- 误差来源分析
- 改进建议（若误差超阈值）""",
    "tools_hint": (
        "推荐工具：astro_dynamics(check_engine_availability, propagate_orbit, "
        "cross_validate_results) | loop_engine（迭代验证+修复）| "
        "RAG(各引擎精度特性文献)"
    ),
}


# ---------------------------------------------------------------------------
# 7. 文献检索
# ---------------------------------------------------------------------------
LITERATURE_SEARCH: Dict[str, str] = {
    "system": """\
【文献检索专属指令】
你的目标是检索最新航天文献，评估相关性，下载强相关论文并总结全文。
检索流程：
  1. 用 literature_search 工具发起 arXiv 搜索（query + research_topic）。
  2. 工具会自动：评估摘要相关性 → 下载 strong 相关 PDF → 全文总结 →
     索引入 RAG → 更新知识图谱。
  3. 检索完成后用 RAG query_enhanced 验证已有结论的证据支撑。
  4. 输出须含：检索统计（总数/强相关/弱相关）、论文清单（标题+作者+相关性）、
     全文总结、知识图谱快照、对当前任务的支撑结论。
  5. 论文全文与 PDF 路径须 offload，上下文只保留标题与摘要引用。
约束：相关性标注 strong/weak；引用须含 arxiv_id；总结须客观不添加原文未述内容。""",
    "user_template": """\
【文献检索任务】
{task}

输入提示：
- 搜索关键词 (query)：如 "lunar transfer orbit low energy"
- 研究主题 (research_topic)：用于相关性评分的主题描述
- 最大结果数 (max_results)：默认 5
- 是否下载强相关 PDF (download_strong)：默认 True

期望输出：
- 检索统计：总数 / 强相关 / 弱相关 / 已下载
- 论文清单：标题 + arxiv_id + 作者 + 相关性 + 全文总结
- 知识图谱快照（节点数 + 边数）
- 对当前任务的支撑结论（结合 RAG 证据验证）
- PDF 路径引用（offload）""",
    "tools_hint": (
        "推荐工具：literature_search | RAG(query_enhanced, query_with_verification) | "
        "CEO Offload(论文全文/PDF) | 可选：generate_knowledge_cloud"
    ),
}


# ---------------------------------------------------------------------------
# 汇总注册表
# ---------------------------------------------------------------------------
TASK_TEMPLATES: Dict[str, Dict[str, str]] = {
    "orbit_design": ORBIT_DESIGN,
    "launch_window": LAUNCH_WINDOW,
    "trajectory_analysis": TRAJECTORY_ANALYSIS,
    "ground_access": GROUND_ACCESS,
    "lunar_transfer": LUNAR_TRANSFER,
    "cross_validation": CROSS_VALIDATION,
    "literature_search": LITERATURE_SEARCH,
}

# 支持的任务类型列表（便于校验与自动补全）
TASK_TYPES: list = list(TASK_TEMPLATES.keys())
