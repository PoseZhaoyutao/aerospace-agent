"""流式输出 — 1:1 复刻 CCB services/api/claude.ts 的 streaming 机制。

真实 SSE 流式：逐 token 输出，不等完整响应。

关键设计（照搬 CCB）：
    1. stream_chat_with_tools() — 真正的流式调用
    2. 逐 chunk yield（不等完整响应）
    3. 流式过程中解析 tool_call 标签
    4. 流式完成后返回完整结构化结果
    5. 支持中断（abort）

Qwen3 API 适配：
    - Qwen3 API server 支持 stream=true 参数
    - 返回 SSE 格式的 data: {...}\n\n
    - 需要解析 delta.content 逐字提取
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional


# ======================================================================
# 流式事件类型
# ======================================================================

@dataclass
class StreamChunk:
    """流式 chunk。"""
    type: str  # "text_delta" | "tool_call_delta" | "done" | "error"
    text: str = ""
    tool_call: Optional[Dict] = None
    error: Optional[str] = None


# ======================================================================
# SSE 解析器
# ======================================================================

def parse_sse_stream(response_bytes: Iterator[bytes]) -> Iterator[StreamChunk]:
    """解析 SSE 流，逐 chunk 产出 StreamChunk。

    对应 CCB 中对 Anthropic SDK Stream 的消费。
    """
    buffer = ""
    for line_bytes in response_bytes:
        line = line_bytes.decode("utf-8", errors="replace")
        buffer += line

        # SSE 事件以双换行分隔
        while "\n\n" in buffer:
            event_str, buffer = buffer.split("\n\n", 1)
            for event_line in event_str.split("\n"):
                event_line = event_line.strip()
                if not event_line.startswith("data: "):
                    continue
                data_str = event_line[6:]
                if data_str == "[DONE]":
                    yield StreamChunk(type="done")
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # 解析 OpenAI 兼容 SSE 格式
                choices = data.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                content = delta.get("content")
                if content:
                    yield StreamChunk(type="text_delta", text=content)

                # vLLM + Qwen3.5: reasoning 字段（thinking 模式）
                reasoning = delta.get("reasoning")
                if reasoning:
                    yield StreamChunk(type="text_delta", text=reasoning)

                # tool_calls（如果 API 支持）
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        yield StreamChunk(type="tool_call_delta", tool_call={
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", ""),
                        })

                if finish_reason:
                    yield StreamChunk(type="done")
                    return


# ======================================================================
# 流式 HTTP 请求
# ======================================================================

def stream_http_request(
    url: str,
    payload: Dict,
    headers: Dict,
    timeout: int = 120,
) -> Iterator[bytes]:
    """发送流式 HTTP 请求，逐行返回响应字节。

    使用 urllib 的流式读取。
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            yield line


# ======================================================================
# 流式 chat_with_tools
# ======================================================================

def stream_chat_with_tools(
    base_url: str,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    api_key: str = "local-no-key-needed",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: int = 120,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None,
) -> Dict:
    """流式 chat_with_tools — 真实 SSE 逐字输出。

    对应 CCB 的 queryModelWithStreaming()。

    流程：
    1. 将 tools 注入 system prompt（Qwen3 <tool_call> 策略）
    2. 发送 stream=true 的请求
    3. 逐 chunk 解析 SSE
    4. 实时调用 on_chunk 回调
    5. 流式完成后解析 <tool_call> 标签
    6. 返回完整结构化结果
    """
    url = base_url.rstrip("/") + "/chat/completions"

    # 1. 构建 messages（注入 tools 到 system prompt）
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    tool_prompt = (
        "你可以使用以下工具完成任务。当需要调用工具时，严格输出以下格式：\n"
        "<tool_call>\n"
        '{"name": "工具名", "arguments": {"参数名": "参数值"}}\n'
        "</tool_call>\n\n"
        "可用工具：\n"
        f"{tools_json}\n\n"
        "规则：\n1. 每次只调用一个工具\n"
        "2. 工具调用后等待结果\n3. 任务完成后直接回答\n"
    )

    adapted_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            adapted_messages.append({
                "role": "system",
                "content": msg["content"] + "\n\n" + tool_prompt,
            })
        else:
            adapted_messages.append(msg)
    if not any(m.get("role") == "system" for m in adapted_messages):
        adapted_messages.insert(0, {"role": "system", "content": tool_prompt})

    # 2. 构建流式请求
    payload = {
        "model": model,
        "messages": adapted_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 3. 流式读取
    full_text = ""
    try:
        for line in stream_http_request(url, payload, headers, timeout):
            for chunk in parse_sse_stream([line]):
                if chunk.type == "text_delta":
                    full_text += chunk.text
                    if on_chunk:
                        on_chunk(chunk)
                elif chunk.type == "done":
                    if on_chunk:
                        on_chunk(chunk)
                elif chunk.type == "error":
                    if on_chunk:
                        on_chunk(chunk)
    except urllib.error.URLError as e:
        return {
            "content": None,
            "tool_calls": None,
            "finish_reason": "error",
            "error": f"Stream connection error: {e}",
        }
    except Exception as e:
        return {
            "content": None,
            "tool_calls": None,
            "finish_reason": "error",
            "error": f"Stream error: {e}",
        }

    # 4. 解析 <tool_call> 标签
    tool_calls = []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, full_text, re.DOTALL)
    for i, match in enumerate(matches):
        try:
            call = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{i}",
                "name": call.get("name", ""),
                "arguments": call.get("arguments", {}),
            })
        except json.JSONDecodeError:
            try:
                call = json.loads(match.strip().rstrip(","))
                tool_calls.append({
                    "id": f"call_{i}",
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", {}),
                })
            except Exception:
                pass

    clean_text = re.sub(pattern, "", full_text, flags=re.DOTALL).strip()

    if tool_calls:
        return {
            "content": clean_text if clean_text else None,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls",
        }
    return {"content": full_text, "tool_calls": None, "finish_reason": "stop"}
