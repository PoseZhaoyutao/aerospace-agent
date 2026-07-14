"""航天意图路由器。

根据用户输入将查询路由到正确的航天领域分支:
    - orbit_design: 轨道设计/参数计算
    - analysis: 轨道分析/传播/机动
    - knowledge: 知识查询/RAG 检索
    - general: 通用对话/工具发现

路由策略:
    1. 关键词匹配（快速路径）
    2. LLM 分类（高精度路径，需要 LLM 可用时）
    3. 默认回退（general）
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .prompts import AEROSPACE_ASSISTANT_IDENTITY

# 航天领域关键词路由表
INTENT_PATTERNS: Dict[str, List[str]] = {
    "orbit_design": [
        r"设计.*轨道", r"轨道.*设计", r"轨道.*参数", r"轨道.*高度",
        r"轨道.*倾角", r"轨道.*类型", r"\b(?:LEO|MEO|GEO|HEO|SSO)\b",
        r"design.*orbit", r"orbit.*design", r"orbit.*param",
        r"轨道根数", r"开普勒", r"kepler",
        r"\d+km.*轨道", r"轨道.*\d+km", r"高度.*\d+.*km",
        r"太阳同步", r"极轨", r"地球同步", r"静止轨道",
        r"设计.*LEO", r"设计.*MEO", r"设计.*GEO",
    ],
    "orbit_propagation": [
        r"轨道.{0,8}传播", r"传播.{0,8}轨道", r"(?:卫星|星历|状态).{0,8}传播",
        r"\b(?:orbit|state|ephemeris)\b.{0,16}\bpropagat\w*",
        r"\bpropagat\w*.{0,16}\b(?:orbit|state|ephemeris)\b",
        r"轨道.*演化", r"轨道.*预测",
        r"数值积分", r"\bRK(?:\d{1,2})?\b", r"积分器",
        r"\bnumerical(?:ly)?\s+integrat\w*",
        r"\b(?:orbit|state|trajectory|equations?)\b.{0,16}\bintegrat\w*",
        r"\bintegrat\w*.{0,16}\b(?:orbit|state|trajectory|equations?)\b",
        r"星历", r"ephemer",
    ],
    "launch_window": [
        r"发射窗口", r"launch.*window", r"发射.*时间",
        r"发射.*时机", r"窗口.*计算",
    ],
    "lunar_transfer": [
        r"月球.*转移", r"地月.*转移", r"lunar.*transfer",
        r"月球.*轨道", r"lunar.*orbit", r"奔月",
        r"Hohmann|霍曼",
    ],
    "maneuver_planning": [
        r"(?:轨道|航天器|卫星).{0,8}机动", r"机动.{0,8}(?:轨道|规划|计算|分析|仿真)",
        r"\b(?:orbit|spacecraft|satellite)\b.{0,16}\bmaneuver\w*",
        r"\bmaneuver\w*.{0,16}\b(?:orbit|spacecraft|satellite)\b",
        r"变轨", r"delta.*v",
        r"速度增量", r"Δv", r"\bdv\b",
    ],
    "knowledge_query": [
        r"(?:什么是|解释|定义|原理|为什么|怎么.*理解).{0,24}(?:轨道|航天|天体|卫星|星历|引力|二体|摄动|动力学)",
        r"(?:轨道|航天|天体|卫星|星历|引力|二体|摄动|动力学).{0,24}(?:是什么|解释|定义|原理|为什么|概念)",
        r"\b(?:what\s+is|explain|define)\b.{0,40}\b(?:orbit|orbital|spacecraft|satellite|astrodynamics|two[-\s]?body|gravity|ephemeris|perturbation)\b",
        r"\b(?:orbit|orbital|spacecraft|satellite|astrodynamics|two[-\s]?body|gravity|ephemeris|perturbation)\b.{0,40}\b(?:what\s+is|explain|definition|principle|concept)\b",
        r"轨道.*力学", r"航天.*力学", r"天体.*力学",
        r"orbital.*mechanics", r"astro.*dynamics",
    ],
    "tool_discovery": [
        r"^\s*工具\s*$",
        r"工具.{0,8}(?:列表|有哪些|可用|能力|功能|状态|发现)",
        r"(?:列出|查看|发现|有哪些).{0,8}(?:工具|引擎|能力|功能)",
        r"(?:你|代理|agent).{0,6}能.*做什么",
        r"^\s*(?:help|tools?)\s*$",
        r"\b(?:list|show|find|discover)\b.{0,24}\b(?:tools?|engines?|capabilities)\b",
        r"\b(?:tools?|engines?)\b.{0,24}\b(?:available|availability|capabilities)\b",
        r"\bwhat\s+can\s+(?:you|this\s+agent)\s+do\b",
        r"check.{0,16}engine",
    ],
}


def classify_intent_keyword(user_message: str) -> Tuple[str, float]:
    """基于关键词匹配的意图分类（快速路径）。

    Args:
        user_message: 用户输入文本

    Returns:
        (intent, confidence): 意图类型和置信度
    """
    msg_lower = user_message.lower()
    scores: Dict[str, int] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general", 0.0

    # 取最高分，平局时按优先级: knowledge_query > orbit_design > 其他
    max_score = max(scores.values())
    candidates = [k for k, v in scores.items() if v == max_score]

    # 优先级排序（knowledge_query 优先，因为疑问词应优先于术语匹配）
    priority_order = [
        "knowledge_query", "tool_discovery",
        "orbit_design", "orbit_propagation",
        "launch_window", "lunar_transfer", "maneuver_planning",
    ]
    best_intent = candidates[0]
    for p in priority_order:
        if p in candidates:
            best_intent = p
            break

    total_score = sum(scores.values())

    # 置信度 = 最高分 / 总分
    confidence = max_score / total_score if total_score > 0 else 0.0

    return best_intent, min(confidence, 1.0)


def classify_intent_llm(
    user_message: str,
    llm,
) -> Tuple[str, float]:
    """基于 LLM 的意图分类（高精度路径）。

    Args:
        user_message: 用户输入
        llm: LLM 接口实例（需有 chat() 方法）

    Returns:
        (intent, confidence): 意图类型和置信度
    """
    prompt = (
        "你是一个航天任务意图分类器。请将以下用户消息分类到以下类别之一，"
        "并给出置信度 (0.0-1.0):\n\n"
        "类别:\n"
        "- orbit_design: 轨道设计、参数计算\n"
        "- orbit_propagation: 轨道传播、数值积分\n"
        "- launch_window: 发射窗口分析\n"
        "- lunar_transfer: 月球/地月转移轨道\n"
        "- maneuver_planning: 轨道机动、变轨\n"
        "- knowledge_query: 航天知识查询、概念解释\n"
        "- tool_discovery: 询问工具有哪些/能做什么\n"
        "- general: 通用对话\n\n"
        f"用户消息: {user_message}\n\n"
        "请只返回 JSON 格式: {\"intent\": \"...\", \"confidence\": 0.X}"
    )

    try:
        try:
            response = llm.chat(
                prompt,
                system_prompt=(
                    f"{AEROSPACE_ASSISTANT_IDENTITY}\n"
                    "你现在只执行意图分发协议，只返回要求的 intent JSON，不要向用户解释。"
                ),
                max_tokens=128,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            response = llm.chat(prompt)
        # 尝试提取 JSON
        import json
        import re

        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
            intent = result.get("intent", "general")
            confidence = float(result.get("confidence", 0.5))
            if intent in INTENT_PATTERNS or intent == "general":
                return intent, min(max(confidence, 0.0), 1.0)
    except Exception:
        pass

    return "general", 0.3


def route_intent(
    user_message: str,
    llm=None,
    use_llm: bool = False,
) -> Tuple[str, float]:
    """路由用户意图。

    优先使用关键词匹配，当置信度 < 0.5 或 use_llm=True 时启用 LLM 增强。

    Args:
        user_message: 用户输入
        llm: LLM 接口（可选）
        use_llm: 是否强制使用 LLM 增强分类

    Returns:
        (intent, confidence)
    """
    keyword_intent, keyword_conf = classify_intent_keyword(user_message)

    # 置信度低于 0.5 时自动启用 LLM（如果可用）
    if (keyword_conf < 0.5 or use_llm) and llm is not None:
        try:
            llm_intent, llm_conf = classify_intent_llm(user_message, llm)

            # 如果 LLM 和关键词一致，提高置信度
            if llm_intent == keyword_intent:
                return llm_intent, max(keyword_conf, llm_conf, 0.5)

            # LLM 置信度高时优先 LLM
            if llm_conf >= 0.6:
                return llm_intent, llm_conf

            # 都不确定时用关键词 + 降低置信度
            return keyword_intent, min(keyword_conf, 0.4)
        except Exception:
            pass

    return keyword_intent, keyword_conf


def get_intent_description(intent: str) -> str:
    """获取意图的人类可读描述。

    Args:
        intent: 意图类型

    Returns:
        中文描述
    """
    descriptions = {
        "orbit_design": "轨道设计与参数计算",
        "orbit_propagation": "轨道传播与数值积分",
        "launch_window": "发射窗口分析",
        "lunar_transfer": "月球转移轨道",
        "maneuver_planning": "轨道机动规划",
        "knowledge_query": "航天知识查询",
        "tool_discovery": "工具发现",
        "general": "通用对话",
    }
    return descriptions.get(intent, "未知意图")
