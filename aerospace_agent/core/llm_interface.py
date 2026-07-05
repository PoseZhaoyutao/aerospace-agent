"""可插拔 LLM 接口模块。

提供统一的 LLM 抽象基类与多种实现，便于在真实 API 与离线回退之间切换：

- ``LLMInterface``        : 抽象基类，定义 ``chat`` / ``stream_chat`` 接口
- ``OpenAICompatibleLLM`` : 通过环境变量调用 OpenAI 兼容 API（标准库实现，支持重试）
- ``MockLLM``             : 基于规则的本地回退，无需 API key，便于离线测试
- ``create_llm``          : 工厂函数，根据环境变量自动选择实现

环境变量约定：
    AEROSPACE_LLM_API_KEY  : API 密钥（存在则使用真实 LLM，否则回退 MockLLM）
    AEROSPACE_LLM_BASE_URL : API 基址（默认 https://api.openai.com/v1）
    AEROSPACE_LLM_MODEL    : 模型名（默认 gpt-3.5-turbo）
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List


class LLMInterface(ABC):
    """LLM 接口抽象基类。

    所有具体 LLM 实现都应继承此类并实现 ``chat`` 方法。
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """同步对话接口。

        Args:
            messages: 消息列表，每个元素形如
                ``{"role": "system"/"user"/"assistant", "content": "..."}``
            **kwargs: 额外参数，如 ``temperature`` / ``max_tokens`` / ``timeout``

        Returns:
            模型回复文本
        """
        raise NotImplementedError

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式对话接口。

        默认实现回退到同步 ``chat``，再按字符块产出，便于上层统一消费。
        子类可覆盖以实现真正的流式调用。
        """
        text = self.chat(messages, **kwargs)
        chunk_size = int(kwargs.get("chunk_size", 24))
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]


class OpenAICompatibleLLM(LLMInterface):
    """OpenAI 兼容 API 客户端（标准库 urllib 实现，不依赖 openai 包）。

    通过环境变量配置，调用 ``{base_url}/chat/completions`` 端点，
    失败时按指数退避重试。
    """

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = None, max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.api_key = api_key or os.environ.get("AEROSPACE_LLM_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("AEROSPACE_LLM_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.model = (
            model
            or os.environ.get("AEROSPACE_LLM_MODEL")
            or "gpt-3.5-turbo"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.api_key:
            raise RuntimeError(
                "缺少 AEROSPACE_LLM_API_KEY 环境变量，无法调用真实 LLM。"
                "请设置该变量，或使用 MockLLM 进行离线测试。"
            )
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST"
                )
                timeout = kwargs.get("timeout", 60)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    obj = json.loads(body)
                    return obj["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                last_err = f"URLError: {e.reason}"
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last_err = f"响应解析失败: {e}"
            # 指数退避等待后重试
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_err}"
        )


class MockLLM(LLMInterface):
    """基于规则的本地回退 LLM。

    当无 API key 时使用，根据用户意图返回简单结构化文本，便于离线测试。
    能识别航天领域常见任务（地月转移、轨道设计、姿态控制、导航、变轨、再入）
    并返回合理的结构化响应；同时支持 ReAct 模式的工具调用输出，使 Agent
    的 think->act->observe 循环可在离线环境下完整演示。
    """

    # 意图关键词 -> 意图标签
    INTENT_RULES = [
        (["地月转移", "月球转移", "lunar transfer", "奔月"], "lunar_transfer"),
        (["轨道设计", "设计轨道", "orbit design", "霍曼", "hohmann"], "orbit_design"),
        (["姿态", "attitude", "姿态控制", "姿态确定"], "attitude"),
        (["导航", "navigation", "定轨", "定位", "星敏"], "navigation"),
        (["变轨", "机动", "maneuver", "delta-v", "速度增量"], "maneuver"),
        (["再入", "reentry", "返回", "entry"], "reentry"),
    ]

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 取最后一条 user 消息作为当前输入
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
        # 汇总 system 消息（包含可用工具与上下文）
        system_text = "".join(
            m.get("content", "") for m in messages if m.get("role") == "system"
        )
        # 判断是否处于 ReAct“观察后”阶段（已有工具返回）
        observed = user_msg.lstrip().startswith("Observation:")
        if not observed:
            observed = any(
                "Observation:" in m.get("content", "") for m in messages
            )
        intent = self._detect_intent(user_msg + " " + system_text)
        return self._generate(intent, user_msg, system_text, observed)

    def _detect_intent(self, text: str) -> str:
        """根据关键词检测任务意图。"""
        low = text.lower()
        for keywords, intent in self.INTENT_RULES:
            for kw in keywords:
                if kw in low or kw in text:
                    return intent
        return "general"

    def _generate(self, intent: str, user_msg: str,
                  system_text: str, observed: bool) -> str:
        """根据意图与对话阶段生成结构化响应。"""
        # 观察后阶段：直接给出最终答案
        if observed and intent in ("lunar_transfer", "orbit_design"):
            return self._final_lunar()
        if observed:
            return self._final_generic(intent)

        # 第一步：若存在相关工具，先发起工具调用以演示 ReAct 循环
        if "orbit_calculator" in system_text and intent in (
            "lunar_transfer", "orbit_design", "maneuver"
        ):
            return (
                "Thought: 用户需要轨道参数，我先调用轨道计算工具获取基础数值，"
                "再综合给出设计方案。\n"
                "Action: orbit_calculator\n"
                'Action Input: {"mission": "lunar_transfer", "altitude_km": 300}\n'
            )

        # 无工具或其它意图：直接给出最终答案
        if intent == "lunar_transfer":
            return self._final_lunar()
        return self._final_generic(intent)

    @staticmethod
    def _final_lunar() -> str:
        """地月转移轨道设计的结构化最终答案。"""
        return (
            "Thought: 综合轨道计算结果与任务需求，给出完整的地月转移轨道设计方案。\n\n"
            "【地月转移轨道设计方案】\n\n"
            "## 1. 任务概述\n"
            "- 任务类型：地月转移轨道（Lunar Transfer Orbit, LTO）\n"
            "- 起始轨道：近地停泊轨道（LEO，约 200~400 km）\n"
            "- 目标：进入月球引力影响球并实现月球轨道插入（LOI）\n\n"
            "## 2. 关键参数（典型值）\n"
            "- C3 能量（双曲线剩余能量）：约 -0.6 ~ -2.0 km²/s²\n"
            "- TLI 速度增量 Δv1：约 3.1~3.2 km/s（由停泊轨道加速）\n"
            "- LOI 速度增量 Δv2：约 0.8~1.0 km/s\n"
            "- 转移时间：约 3~5 天（Hohmann 型转移）\n"
            "- 近地点速度（300km 停泊轨道）：约 7.73 km/s\n\n"
            "## 3. 设计步骤\n"
            "1. 发射窗口分析（月球黄经位置匹配）\n"
            "2. 停泊轨道参数确定（高度、倾角）\n"
            "3. 地月转移能量需求 C3 计算\n"
            "4. TLI 机动设计（速度增量大小与方向）\n"
            "5. 中途修正机动（TCM）规划，通常 2~3 次\n"
            "6. LOI 机动设计（捕获至月球轨道）\n\n"
            "## 4. 推荐工具\n"
            "- orbit_calculator：基础轨道参数计算\n"
            "- Lambert 问题求解器（如可用）\n"
            "- 发射窗口分析工具\n\n"
            "Final Answer: 已生成地月转移轨道设计方案，包含 C3 能量、TLI/LOI 速度增量、"
            "转移时间等关键参数及 6 步设计流程。300 km 停泊轨道下 TLI 约 3.1 km/s，"
            "转移时间约 5 天。如需精确数值可进一步调用轨道力学工具求解。"
        )

    @staticmethod
    def _final_generic(intent: str) -> str:
        """通用意图的结构化最终答案。"""
        templates = {
            "orbit_design": "已完成轨道设计分析：建议先确定约束（半长轴、偏心率、"
                            "倾角），再通过 Lambert 或 Hohmann 方法求解转移参数。",
            "attitude": "已完成姿态控制分析：建议基于四元数描述姿态，"
                        "采用 PD 或 LQR 控制律，注意避免欧拉角奇异。",
            "navigation": "已完成导航分析：建议采用扩展卡尔曼滤波（EKF）"
                          "融合星敏感器与惯性测量单元数据。",
            "maneuver": "已完成变轨机动分析：根据速度增量需求选择脉冲机动或"
                        "有限推力机动，注意摄动修正。",
            "reentry": "已完成再入分析：建议采用跳跃式再入剖面以降低过载，"
                       "关注再入倾角与航程的权衡。",
        }
        body = templates.get(intent, "已根据任务需求完成分析并给出建议。")
        return (
            "Thought: 根据任务意图给出结构化分析结论。\n\n"
            f"【任务分析】\n{body}\n\n"
            "Final Answer: " + body
        )


def create_llm(api_key: str = None, base_url: str = None,
               model: str = None, force_mock: bool = False,
               **kwargs) -> LLMInterface:
    """工厂函数：根据环境变量自动选择 LLM 实现。

    优先级：
      1. 若 ``force_mock=True``，强制使用 MockLLM；
      2. 若存在 ``AEROSPACE_LLM_API_KEY``（或显式传入 api_key），使用
         ``OpenAICompatibleLLM``；
      3. 否则使用 ``MockLLM``（离线回退）。
    """
    if force_mock:
        return MockLLM()
    api_key = api_key or os.environ.get("AEROSPACE_LLM_API_KEY")
    if api_key:
        return OpenAICompatibleLLM(
            api_key=api_key, base_url=base_url, model=model, **kwargs
        )
    return MockLLM()
