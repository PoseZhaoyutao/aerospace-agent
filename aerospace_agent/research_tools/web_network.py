"""网络与 Web 工具集 (10 个)——仅依赖标准库 urllib。

设计原则：
  1. 全部使用 ``urllib`` 标准库，不依赖 requests 等第三方库
  2. 网络不可用时优雅返回 ``{"status":"error","reason":"network unavailable"}``
  3. 所有工具返回 dict / list / str（JSON 可序列化）
  4. 统一超时控制，避免阻塞 Agent 主循环

工具清单
--------
- http_get          : HTTP GET 请求
- http_post         : HTTP POST 请求
- download_file     : 从 URL 下载文件到本地
- web_search        : 搜索 web（返回 URL 列表）
- fetch_page        : 获取网页内容（返回文本）
- parse_html_text   : 从 HTML 提取纯文本
- api_call          : 通用 API 调用（GET/POST，支持 headers 和 params）
- check_url         : 检查 URL 可访问性（返回状态码）
- read_json_api     : 获取 JSON API 并解析
- url_encode_decode : URL 编解码
"""
from __future__ import annotations

import json as _json
import re
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from aerospace_agent.research_tools.base import register_tool

# 默认 User-Agent，避免被部分站点拦截
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AerospaceAgent/ResearchTool"
)

# 视为"网络不可用"的异常类型
_NETWORK_ERRORS = (
    urllib_error.URLError,
    urllib_error.HTTPError,
    ConnectionError,
    TimeoutError,
    OSError,
    ValueError,
)


def _network_unavailable() -> Dict[str, Any]:
    """统一的网络不可用错误返回。"""
    return {"status": "error", "reason": "network unavailable"}


def _build_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """合并自定义 headers 与默认 User-Agent。"""
    h = {"User-Agent": _DEFAULT_UA}
    if headers:
        h.update(headers)
    return h


def _append_params(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    """将 params 追加到 URL query string。"""
    if not params:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + urllib_parse.urlencode(params)


@register_tool(
    "http_get",
    "HTTP GET 请求，返回状态码、响应头与正文",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "目标 URL"},
        {"name": "headers", "type": "dict", "description": "自定义请求头",
         "required": False, "default": {}},
        {"name": "params", "type": "dict", "description": "URL 查询参数",
         "required": False, "default": {}},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 30},
    ],
)
def http_get(url, headers=None, params=None, timeout=30):
    full_url = _append_params(url, params)
    req = urllib_request.Request(full_url, headers=_build_headers(headers))
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")
            return {
                "status": "success",
                "url": full_url,
                "status_code": resp.status,
                "headers": dict(resp.headers),
                "body": text,
            }
    except _NETWORK_ERRORS:
        return _network_unavailable()


@register_tool(
    "http_post",
    "HTTP POST 请求，支持表单或 JSON body",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "目标 URL"},
        {"name": "data", "type": "dict", "description": "POST 数据",
         "required": False, "default": {}},
        {"name": "headers", "type": "dict", "description": "自定义请求头",
         "required": False, "default": {}},
        {"name": "json_mode", "type": "bool",
         "description": "True 以 JSON 发送，False 以表单发送",
         "required": False, "default": True},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 30},
    ],
)
def http_post(url, data=None, headers=None, json_mode=True, timeout=30):
    data = data or {}
    merged_headers = _build_headers(headers)
    if json_mode:
        payload = _json.dumps(data).encode("utf-8")
        merged_headers.setdefault("Content-Type", "application/json")
    else:
        payload = urllib_parse.urlencode(data).encode("utf-8")
        merged_headers.setdefault(
            "Content-Type", "application/x-www-form-urlencoded"
        )
    req = urllib_request.Request(url, data=payload, headers=merged_headers,
                                 method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            text = body.decode(charset, errors="replace")
            return {
                "status": "success",
                "url": url,
                "status_code": resp.status,
                "headers": dict(resp.headers),
                "body": text,
            }
    except _NETWORK_ERRORS:
        return _network_unavailable()


@register_tool(
    "download_file",
    "从 URL 下载文件到本地路径",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "文件下载 URL"},
        {"name": "path", "type": "str", "description": "本地保存路径"},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 60},
        {"name": "headers", "type": "dict", "description": "自定义请求头",
         "required": False, "default": {}},
    ],
)
def download_file(url, path, timeout=60, headers=None):
    req = urllib_request.Request(url, headers=_build_headers(headers))
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return {
            "status": "success",
            "url": url,
            "path": path,
            "bytes": len(data),
        }
    except _NETWORK_ERRORS:
        return _network_unavailable()
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool(
    "web_search",
    "搜索 web 并返回结果列表（标题 + URL），使用 urllib 简单解析",
    "web_network",
    params=[
        {"name": "query", "type": "str", "description": "搜索关键词"},
        {"name": "max_results", "type": "int", "description": "最大返回条数",
         "required": False, "default": 10},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 30},
    ],
)
def web_search(query, max_results=10, timeout=30):
    # 使用 DuckDuckGo HTML 端点（无需 API key）
    endpoint = "https://html.duckduckgo.com/html/"
    data = urllib_parse.urlencode({"q": query}).encode("utf-8")
    req = urllib_request.Request(
        endpoint, data=data, headers=_build_headers(), method="POST"
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            html = resp.read().decode(charset, errors="replace")
    except _NETWORK_ERRORS:
        return _network_unavailable()

    results: List[Dict[str, str]] = []
    # DuckDuckGo HTML 结果中的链接形如 <a class="result__a" href="...">title</a>
    # href 常含跳转参数 uddg=，需解析真实 URL
    link_re = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for raw_href, raw_title in link_re.findall(html):
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        if not title:
            continue
        # 解析 uddg 参数得到真实 URL
        parsed = urllib_parse.urlparse(raw_href)
        qs = urllib_parse.parse_qs(parsed.query)
        real_url = qs.get("uddg", [raw_href])[0]
        results.append({"title": title, "url": real_url})
        if len(results) >= max_results:
            break

    return {
        "status": "success",
        "query": query,
        "count": len(results),
        "results": results,
    }


@register_tool(
    "fetch_page",
    "获取网页内容并返回纯文本",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "目标网页 URL"},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 30},
        {"name": "headers", "type": "dict", "description": "自定义请求头",
         "required": False, "default": {}},
    ],
)
def fetch_page(url, timeout=30, headers=None):
    req = urllib_request.Request(url, headers=_build_headers(headers))
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            html = resp.read().decode(charset, errors="replace")
        return {
            "status": "success",
            "url": url,
            "text": html,
            "length": len(html),
        }
    except _NETWORK_ERRORS:
        return _network_unavailable()


@register_tool(
    "parse_html_text",
    "从 HTML 字符串提取纯文本（去除标签与脚本）",
    "web_network",
    params=[
        {"name": "html", "type": "str", "description": "HTML 字符串"},
        {"name": "keep_scripts", "type": "bool",
         "description": "是否保留 script/style 内容", "required": False,
         "default": False},
    ],
)
def parse_html_text(html, keep_scripts=False):
    if not isinstance(html, str):
        return {"status": "error", "reason": "html 必须为字符串"}
    text = html
    if not keep_scripts:
        # 移除 script / style 块
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>",
                      "", text, flags=re.IGNORECASE | re.DOTALL)
    # 移除 HTML 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # 将 <br> / 块级标签转为换行
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", text,
                  flags=re.IGNORECASE)
    # 去除所有剩余标签
    text = re.sub(r"<[^>]+>", "", text)
    # 解码常见 HTML 实体
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    # 折叠多余空白
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return {
        "status": "success",
        "text": text,
        "length": len(text),
        "line_count": text.count("\n") + 1 if text else 0,
    }


@register_tool(
    "api_call",
    "通用 API 调用（GET/POST，支持 headers 和 params）",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "API 端点 URL"},
        {"name": "method", "type": "str",
         "description": "HTTP 方法（GET/POST）", "required": False,
         "default": "GET"},
        {"name": "headers", "type": "dict", "description": "请求头",
         "required": False, "default": {}},
        {"name": "params", "type": "dict", "description": "查询参数",
         "required": False, "default": {}},
        {"name": "data", "type": "dict", "description": "POST body",
         "required": False, "default": {}},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 30},
    ],
)
def api_call(url, method="GET", headers=None, params=None, data=None,
             timeout=30):
    method = (method or "GET").upper()
    full_url = _append_params(url, params)
    merged_headers = _build_headers(headers)
    payload = None
    if method == "POST":
        body = data or {}
        payload = _json.dumps(body).encode("utf-8")
        merged_headers.setdefault("Content-Type", "application/json")
    req = urllib_request.Request(full_url, data=payload,
                                 headers=merged_headers, method=method)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            body_text = raw.decode(charset, errors="replace")
            # 尝试解析为 JSON
            try:
                body_parsed = _json.loads(body_text)
            except (ValueError, TypeError):
                body_parsed = body_text
            return {
                "status": "success",
                "url": full_url,
                "method": method,
                "status_code": resp.status,
                "headers": dict(resp.headers),
                "body": body_parsed,
            }
    except _NETWORK_ERRORS:
        return _network_unavailable()


@register_tool(
    "check_url",
    "检查 URL 可访问性，返回状态码",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "待检查 URL"},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 15},
        {"name": "method", "type": "str",
         "description": "HTTP 方法（默认 HEAD）", "required": False,
         "default": "HEAD"},
    ],
)
def check_url(url, timeout=15, method="HEAD"):
    method = (method or "HEAD").upper()
    req = urllib_request.Request(url, headers=_build_headers(), method=method)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return {
                "status": "success",
                "url": url,
                "status_code": resp.status,
                "accessible": True,
                "reason": resp.reason,
            }
    except urllib_error.HTTPError as e:
        # HTTP 错误码（如 404）也算"可访问"，只是状态码非 2xx
        return {
            "status": "success",
            "url": url,
            "status_code": e.code,
            "accessible": True,
            "reason": e.reason,
        }
    except _NETWORK_ERRORS:
        return {
            "status": "success",
            "url": url,
            "status_code": None,
            "accessible": False,
            "reason": "network unavailable",
        }


@register_tool(
    "read_json_api",
    "获取 JSON API 并解析为 Python 对象",
    "web_network",
    params=[
        {"name": "url", "type": "str", "description": "JSON API URL"},
        {"name": "params", "type": "dict", "description": "查询参数",
         "required": False, "default": {}},
        {"name": "headers", "type": "dict", "description": "请求头",
         "required": False, "default": {}},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 30},
    ],
)
def read_json_api(url, params=None, headers=None, timeout=30):
    full_url = _append_params(url, params)
    req = urllib_request.Request(full_url, headers=_build_headers(headers))
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
        try:
            data = _json.loads(text)
        except (ValueError, TypeError) as e:
            return {
                "status": "error",
                "reason": f"响应不是有效 JSON: {e}",
                "url": full_url,
                "raw": text[:1000],
            }
        return {
            "status": "success",
            "url": full_url,
            "status_code": resp.status,
            "json": data,
        }
    except _NETWORK_ERRORS:
        return _network_unavailable()


@register_tool(
    "url_encode_decode",
    "URL 编码与解码",
    "web_network",
    params=[
        {"name": "text", "type": "str", "description": "待处理文本"},
        {"name": "action", "type": "str",
         "description": "操作：encode 或 decode", "required": False,
         "default": "encode"},
        {"name": "safe", "type": "str",
         "description": "encode 时不编码的字符", "required": False,
         "default": ""},
    ],
)
def url_encode_decode(text, action="encode", safe=""):
    if not isinstance(text, str):
        return {"status": "error", "reason": "text 必须为字符串"}
    action = (action or "encode").lower()
    if action == "encode":
        result = urllib_parse.quote(text, safe=safe)
    elif action == "decode":
        try:
            result = urllib_parse.unquote(text)
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    else:
        return {"status": "error",
                "reason": f"未知 action: {action}（应为 encode 或 decode）"}
    return {
        "status": "success",
        "action": action,
        "input": text,
        "result": result,
    }


# ---- 模块自测 ----
if __name__ == "__main__":
    # 离线可验证的工具
    print("url_encode_decode encode:",
          url_encode_decode("hello world&foo=bar", action="encode"))
    print("url_encode_decode decode:",
          url_encode_decode("hello%20world%26foo%3Dbar", action="decode"))
    print("parse_html_text:",
          parse_html_text("<html><body><p>Hello</p><script>x</script></body></html>"))
    print("check_url (离线):", check_url("https://example.invalid", timeout=3))
