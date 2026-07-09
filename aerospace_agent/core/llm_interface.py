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

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict],
                        **kwargs) -> Dict:
        """原生 Function Calling 接口——LLM 直接输出结构化 tool_call。

        参考 Claude Code 架构：LLM 不再输出文本让 Agent 正则解析，
        而是直接返回 tool_calls 结构化数据。

        Args:
            messages: 消息列表（支持 tool role）
            tools: OpenAI 格式工具定义列表
                [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]

        Returns:
            {"content": str|None, "tool_calls": list|None, "finish_reason": str}
            tool_calls: [{"id":"call_xxx","name":"tool_name","arguments": dict}]
        """
        # 默认回退：子类未实现时用纯文本 chat + 正则解析
        text = self.chat(messages, **kwargs)
        return {"content": text, "tool_calls": None, "finish_reason": "stop"}

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
                    # K5-缺陷11: content 可能为 None（tool_calls / finish_reason 非 stop）
                    content = obj["choices"][0]["message"].get("content")
                    return content or ""
            except urllib.error.HTTPError as e:
                # K5-缺陷10: 4xx 客户端错误（401/403/400）不重试
                if e.code in (400, 401, 403):
                    raise RuntimeError(f"LLM 认证/请求错误 HTTP {e.code}: {e.reason}") from e
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


class LocalLLM(LLMInterface):
    """本地部署小模型接口（OpenAI 兼容端点）。

    支持通过 Ollama / vLLM / llama.cpp server / LM Studio 等本地部署的
    OpenAI 兼容 API 端点调用本地小模型。

    环境变量约定：
        AEROSPACE_LOCAL_LLM_BASE_URL : 本地 API 基址（如 http://localhost:11434/v1）
        AEROSPACE_LOCAL_LLM_MODEL    : 本地模型名（如 qwen2.5:7b / llama3:8b）
        AEROSPACE_LOCAL_LLM_API_KEY  : 本地 API key（Ollama 不需要，vLLM 可选）

    与 OpenAICompatibleLLM 的区别：
        - 默认无 API key 校验（本地模型通常无需鉴权）
        - 超时更长（本地模型推理可能较慢）
        - 标记 ``is_local=True`` 供 ModelRouter 路由
    """

    def __init__(self, base_url: str = None, model: str = None,
                 api_key: str = None, max_retries: int = 2,
                 retry_delay: float = 2.0, timeout: float = None):
        self.base_url = (
            base_url
            or os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL")
            or "http://localhost:11434/v1"
        )
        self.model = (
            model
            or os.environ.get("AEROSPACE_LOCAL_LLM_MODEL")
            or "qwen2.5:7b"
        )
        self.api_key = (
            api_key
            or os.environ.get("AEROSPACE_LOCAL_LLM_API_KEY")
            or "local-no-key-needed"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.is_local = True
        # 实例级默认超时 (秒); None 时回退到 chat() 的 120s 默认值。
        # 慢速本地模型 (如 8B VL 模型) 推理可能需数分钟, 可在构造时
        # 传入更长超时, 例如 ``LocalLLM(..., timeout=600)``。
        self.default_timeout = timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
        }
        # 默认 max_tokens：本地部署窗口通常较小，默认 1024 给 prompt 留空间
        # 可通过 kwargs 或环境变量 AEROSPACE_MAX_OUTPUT_TOKENS 覆盖
        import os as _os
        _default_out = int(_os.environ.get("AEROSPACE_MAX_OUTPUT_TOKENS", 1024))
        max_tokens = kwargs.get("max_tokens", _default_out)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
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
                timeout = kwargs.get("timeout", self.default_timeout or 120)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    obj = json.loads(body)
                    msg = obj["choices"][0]["message"]
                    # vLLM + Qwen3.5: content 可能为 None，推理在 reasoning 字段
                    content = msg.get("content")
                    if content:
                        return content
                    # 回退：提取 reasoning 字段（vLLM thinking 模式）
                    reasoning = msg.get("reasoning", "")
                    if reasoning:
                        return reasoning
                    # 最终回退：空字符串
                    return ""
            except Exception as e:
                last_err = str(e)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"本地 LLM 调用失败（已重试 {self.max_retries} 次）: {last_err}\n"
            f"请确认本地模型服务已启动: {self.base_url}"
        )

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict],
                        **kwargs) -> Dict:
        """原生 Function Calling——Qwen3 ChatML 格式。

        Qwen3 自定义 API 服务器不支持 OpenAI tools 参数，
        但 Qwen3 模型本身知道输出 <tool_call> 标签。
        策略：将 tools 注入 system prompt + 解析 <tool_call> 标签。

        参考: Claude Code 的 agentic loop——LLM 输出结构化 tool_call，
        不依赖正则解析 Action/Action Input 文本。
        """
        import re as _re

        # 1. 将 tools 定义注入 system prompt
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        tool_prompt = (
            "你可以使用以下工具完成任务。当需要调用工具时，严格输出以下格式：\n"
            "<tool_call>\n"
            '{"name": "工具名", "arguments": {"参数名": "参数值"}}\n'
            "</tool_call>\n\n"
            "可用工具：\n"
            f"{tools_json}\n\n"
            "规则：\n"
            "1. 每次只调用一个工具\n"
            "2. 工具调用后等待结果，不要自己编造结果\n"
            "3. 任务完成后直接回答，不再输出 <tool_call>\n"
        )

        # 合并到 system 消息
        adapted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                adapted_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + tool_prompt,
                })
            else:
                adapted_messages.append(msg)

        if not any(m["role"] == "system" for m in adapted_messages):
            adapted_messages.insert(0, {"role": "system", "content": tool_prompt})

        # 2. 调用普通 chat
        try:
            text = self.chat(adapted_messages, **kwargs)
        except Exception as e:
            return {"content": None, "tool_calls": None,
                    "finish_reason": "error", "error": str(e)}

        # 3. 解析 <tool_call> 标签
        tool_calls = []
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = _re.findall(pattern, text, _re.DOTALL)

        for i, match in enumerate(matches):
            try:
                call = json.loads(match.strip())
                tool_calls.append({
                    "id": f"call_{i}",
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", {}),
                })
            except json.JSONDecodeError:
                # 尝试提取 name 和 arguments
                try:
                    call = json.loads(match.strip().rstrip(","))
                    tool_calls.append({
                        "id": f"call_{i}",
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments", {}),
                    })
                except Exception:
                    pass

        # 处理未闭合的 <tool_call> 标签（LLM 输出被 max_tokens 截断）
        if not matches:
            unclosed_pattern = r"<tool_call>\s*(.*)"
            unclosed_match = _re.search(unclosed_pattern, text, _re.DOTALL)
            if unclosed_match:
                raw_json = unclosed_match.group(1).strip()
                # 尝试逐步截断解析（处理截断的 JSON）
                for end in range(len(raw_json), 0, -1):
                    try:
                        call = json.loads(raw_json[:end])
                        tool_calls.append({
                            "id": "call_0",
                            "name": call.get("name", ""),
                            "arguments": call.get("arguments", {}),
                        })
                        break
                    except json.JSONDecodeError:
                        continue

        # 去除 <tool_call> 标签后的纯文本（含未闭合标签）
        clean_text = _re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=_re.DOTALL).strip()

        # 过滤 normalize_messages_for_api 产生的 [Tool Call: ...] 格式
        # LLM thinking 模式可能回显对话历史中的这种格式
        tool_call_ref_pattern = _re.compile(r"\[Tool Call:\s*\w+\]\([^)]*\)")
        clean_text = tool_call_ref_pattern.sub("", clean_text).strip()

        # 同样过滤 [Tool Result] 和 [Tool Error] 格式
        tool_result_ref_pattern = _re.compile(r"\[Tool (?:Result|Error)\]\s*[^:]+:\s*.*", _re.DOTALL)
        clean_text = tool_result_ref_pattern.sub("", clean_text).strip()

        # 去除多余的空行
        clean_text = _re.sub(r"\n{3,}", "\n\n", clean_text).strip()

        if tool_calls:
            return {
                "content": clean_text if clean_text else None,
                "tool_calls": tool_calls,
                "finish_reason": "tool_calls",
            }
        return {
            "content": clean_text if clean_text else None,
            "tool_calls": None,
            "finish_reason": "stop",
        }

    def is_available(self) -> bool:
        """检测本地模型服务是否可达。"""
        try:
            url = self.base_url.rstrip("/") + "/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def stream_chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        on_chunk=None,
        **kwargs,
    ) -> Dict:
        """流式 chat_with_tools — 真实 SSE 逐字输出。

        对应 CCB 的 queryModelWithStreaming()。

        与 chat_with_tools 的区别：
        - 使用 stream=true 参数，逐 token 返回
        - 通过 on_chunk 回调实时输出
        - 流式完成后解析 <tool_call> 标签
        """
        from .streaming import stream_chat_with_tools as _stream_fn
        result = _stream_fn(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            tools=tools,
            api_key=self.api_key,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 1024),
            timeout=kwargs.get("timeout", 120),
            on_chunk=on_chunk,
        )
        return result

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """流式聊天——逐 token 返回生成内容（SSE）。

        K5-缺陷4: 重命名为 stream_chat 以覆盖基类抽象方法，
        使 agent.py 中的 self.llm.stream_chat() 能正确调用此实现。

        用法::

            for chunk in llm.stream_chat(messages):
                print(chunk, end="", flush=True)

        Yields:
            str: 每次产出的文本片段
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "stream": True,
        }
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )
        timeout = kwargs.get("timeout", 300)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line_data = line[6:]
                    if line_data == b"[DONE]":
                        return
                    try:
                        obj = json.loads(line_data)
                        delta = obj.get("choices", [{}])[0].get(
                            "delta", {}
                        )
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError):
                        pass


class ModelRouter:
    """模型路由器——按任务复杂度路由到本地小模型或云端大模型。

    第一性原理（K4）：
        - 简单任务（工具调用、格式转换、短文本生成）→ 本地小模型（低成本低延迟）
        - 复杂推理（第一性原理分析、工作流生成、故障诊断）→ 云端大模型（高质量）
        - 离线场景 → 本地模型或 MockLLM

    路由策略：
        1. 若本地模型可用且任务为 simple → LocalLLM
        2. 若云端 API 可用且任务为 complex → OpenAICompatibleLLM
        3. 若仅本地可用 → LocalLLM（所有任务）
        4. 若均不可用 → MockLLM（离线回退）
    """

    # 简单任务关键词（路由到本地小模型）
    SIMPLE_TASK_KEYWORDS = [
        "convert", "transform", "format", "parse", "extract",
        "list", "search", "query", "check", "status",
        "转换", "查询", "检查", "列表", "格式化", "解析",
    ]
    # 复杂任务关键词（路由到云端大模型）
    COMPLEX_TASK_KEYWORDS = [
        "design", "analyze", "reason", "plan", "diagnose",
        "generate", "optimize", "validate", "fix", "synthesize",
        "设计", "分析", "推理", "规划", "诊断", "生成", "优化", "验证", "修复",
    ]

    def __init__(self, local_llm: LocalLLM = None,
                 cloud_llm: OpenAICompatibleLLM = None,
                 mock_llm: MockLLM = None):
        self.local_llm = local_llm or LocalLLM()
        self.cloud_llm = cloud_llm
        self.mock_llm = mock_llm or MockLLM()

    def classify_task(self, task: str) -> str:
        """分类任务复杂度：simple / complex / unknown。"""
        low = task.lower()
        simple_score = sum(1 for kw in self.SIMPLE_TASK_KEYWORDS if kw in low)
        complex_score = sum(1 for kw in self.COMPLEX_TASK_KEYWORDS if kw in low)
        if complex_score > simple_score:
            return "complex"
        if simple_score > 0:
            return "simple"
        return "unknown"

    def route(self, task: str) -> LLMInterface:
        """根据任务复杂度路由到合适的 LLM。"""
        complexity = self.classify_task(task)
        local_ok = self.local_llm.is_available()
        cloud_ok = self.cloud_llm is not None and bool(
            getattr(self.cloud_llm, "api_key", None))

        if complexity == "simple" and local_ok:
            return self.local_llm
        if complexity == "complex" and cloud_ok:
            return self.cloud_llm
        if local_ok:
            return self.local_llm
        if cloud_ok:
            return self.cloud_llm
        return self.mock_llm

    def chat(self, messages: List[Dict[str, str]], task_hint: str = "",
             **kwargs) -> str:
        """带任务提示的路由对话。"""
        llm = self.route(task_hint or messages[-1].get("content", ""))
        return llm.chat(messages, **kwargs)


def create_llm(api_key: str = None, base_url: str = None,
               model: str = None, force_mock: bool = False,
               use_local: bool = False, use_router: bool = False,
               **kwargs) -> LLMInterface:
    """工厂函数：根据环境变量自动选择 LLM 实现。

    优先级：
      1. 若 ``force_mock=True``，强制使用 MockLLM；
      2. 若 ``use_router=True``，返回 ModelRouter（本地+云端自动路由）；
      3. 若 ``use_local=True`` 或设置 ``AEROSPACE_LOCAL_LLM_BASE_URL``，使用 LocalLLM；
      4. 若存在 ``AEROSPACE_LLM_API_KEY``（或显式传入 api_key），使用
         ``OpenAICompatibleLLM``；
      5. 否则使用 ``MockLLM``（离线回退）。
    """
    if force_mock:
        return MockLLM()
    if use_router:
        local = LocalLLM()
        cloud_api = api_key or os.environ.get("AEROSPACE_LLM_API_KEY")
        cloud = OpenAICompatibleLLM(
            api_key=cloud_api, base_url=base_url, model=model, **kwargs
        ) if cloud_api else None
        return ModelRouter(local_llm=local, cloud_llm=cloud)
    local_url = os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL")
    if use_local or local_url:
        return LocalLLM(base_url=base_url, model=model, **kwargs)
    api_key = api_key or os.environ.get("AEROSPACE_LLM_API_KEY")
    if api_key:
        return OpenAICompatibleLLM(
            api_key=api_key, base_url=base_url, model=model, **kwargs
        )
    return MockLLM()
