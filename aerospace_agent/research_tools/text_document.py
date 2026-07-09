"""文本与文档处理工具集（10 个原子工具）。

第一性原理：
  1. 文本是科研文档的最小语义载体（正则/模板/统计/提取）
  2. 标准库实现（re/string/collections/math），零第三方依赖
  3. 所有返回值 JSON 可序列化——可链式组合
  4. 统一错误协议：失败时返回 {"status":"error","reason":"..."}
"""
from __future__ import annotations

import re
import math
from collections import Counter

from aerospace_agent.research_tools.base import register_tool

# 简易中英文停用词表（基于词频提取关键词时过滤）
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in",
    "on", "at", "by", "for", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "from",
    "has", "have", "had", "not", "no", "do", "does", "did", "will", "would",
    "can", "could", "should", "shall", "may", "might", "must", "i", "you",
    "he", "she", "we", "they", "them", "his", "her", "our", "their",
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "与", "及", "或", "等", "为", "以", "于",
}


@register_tool("regex_search", "正则搜索文本", "text_document",
               params=[{"name": "text", "type": "str", "description": "待搜索文本"},
                       {"name": "pattern", "type": "str", "description": "正则表达式"},
                       {"name": "flags", "type": "str", "description": "标志（如 'i' 忽略大小写, 'm' 多行, 's' 点匹配换行）",
                        "required": False, "default": ""}])
def regex_search(text, pattern, flags=""):
    """正则搜索，返回所有匹配。"""
    try:
        flag = 0
        if "i" in flags:
            flag |= re.IGNORECASE
        if "m" in flags:
            flag |= re.MULTILINE
        if "s" in flags:
            flag |= re.DOTALL
        matches = []
        for m in re.finditer(pattern, text, flag):
            matches.append({
                "match": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "groups": list(m.groups()),
            })
        return {"status": "success", "count": len(matches), "matches": matches}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("regex_replace", "正则替换文本", "text_document",
               params=[{"name": "text", "type": "str", "description": "原始文本"},
                       {"name": "pattern", "type": "str", "description": "正则表达式"},
                       {"name": "replacement", "type": "str", "description": "替换字符串（支持 \\1 反向引用）"},
                       {"name": "count", "type": "int", "description": "替换次数（0 表示全部）",
                        "required": False, "default": 0},
                       {"name": "flags", "type": "str", "description": "标志（i/m/s）",
                        "required": False, "default": ""}])
def regex_replace(text, pattern, replacement, count=0, flags=""):
    """正则替换文本。"""
    try:
        flag = 0
        if "i" in flags:
            flag |= re.IGNORECASE
        if "m" in flags:
            flag |= re.MULTILINE
        if "s" in flags:
            flag |= re.DOTALL
        result = re.sub(pattern, replacement, text, count=count, flags=flag)
        replaced = text.count(pattern) if count == 0 else min(count, text.count(pattern))
        return {"status": "success", "result": result,
                "changed": result != text}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("text_split", "按分隔符分割文本", "text_document",
               params=[{"name": "text", "type": "str", "description": "待分割文本"},
                       {"name": "delimiter", "type": "str", "description": "分隔符（默认空白）",
                        "required": False, "default": ""},
                       {"name": "max_split", "type": "int", "description": "最大分割次数（-1 表示不限）",
                        "required": False, "default": -1},
                       {"name": "strip", "type": "bool", "description": "是否去除每段首尾空白",
                        "required": False, "default": True}])
def text_split(text, delimiter="", max_split=-1, strip=True):
    """按分隔符分割文本。"""
    try:
        if delimiter == "":
            parts = text.split(maxsplit=max_split if max_split != -1 else None)
        else:
            parts = text.split(delimiter, max_split if max_split != -1 else -1)
        if strip:
            parts = [p.strip() for p in parts]
        parts = [p for p in parts if p != ""] if strip else parts
        return {"status": "success", "parts": parts, "count": len(parts)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("text_join", "用分隔符连接文本列表", "text_document",
               params=[{"name": "items", "type": "list", "description": "文本列表"},
                       {"name": "delimiter", "type": "str", "description": "分隔符",
                        "required": False, "default": ","}])
def text_join(items, delimiter=","):
    """用分隔符连接文本列表（非字符串元素自动转为字符串）。"""
    try:
        result = delimiter.join(str(i) for i in items)
        return {"status": "success", "result": result, "count": len(items)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("template_render", "简单模板渲染（{var} 格式）", "text_document",
               params=[{"name": "template", "type": "str", "description": "模板字符串，含 {var} 占位符"},
                       {"name": "variables", "type": "dict", "description": "变量名到值的映射"}])
def template_render(template, variables):
    """简单模板渲染，用 variables 填充 {var} 占位符。"""
    try:
        # 使用 str.format_map，缺失键保留原占位符
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        result = template.format_map(_SafeDict(variables))
        return {"status": "success", "result": result,
                "variables_used": len(variables)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("extract_tables", "从文本提取表格", "text_document",
               params=[{"name": "text", "type": "str", "description": "包含表格的文本"},
                       {"name": "format", "type": "str", "description": "表格格式 (markdown/pipe/csv/auto)",
                        "required": False, "default": "auto"}])
def extract_tables(text, format="auto"):
    """从文本提取表格（支持 markdown 管道表格与 csv 风格）。"""
    try:
        lines = text.splitlines()
        tables = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # 管道表格：行中含 |
            if "|" in line and line.count("|") >= 2:
                block = []
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    block.append(lines[i].strip())
                    i += 1
                table = _parse_pipe_table(block)
                if table:
                    tables.append(table)
                continue
            # 连续多行包含相同数量的逗号 -> csv 风格表格
            if format in ("csv", "auto") and "," in line:
                block = []
                ncol = line.count(",") + 1
                while (i < len(lines) and lines[i].strip()
                       and lines[i].count(",") + 1 == ncol):
                    block.append(lines[i].strip())
                    i += 1
                if len(block) >= 2:
                    table = _parse_csv_table(block)
                    if table:
                        tables.append(table)
                    continue
            i += 1
        return {"status": "success", "tables": tables, "count": len(tables)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def _parse_pipe_table(block):
    """解析管道表格块。"""
    rows = []
    for ln in block:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    # 跳过分隔行（如 ---|---）
    if len(rows) >= 2 and all(re.fullmatch(r"[\s\-:]+", c) for c in rows[1]):
        header, body = rows[0], rows[2:]
    else:
        header, body = (rows[0] if rows else []), rows[1:]
    data = [dict(zip(header, r)) for r in body]
    return {"header": header, "rows": data, "row_count": len(data)}


def _parse_csv_table(block):
    """解析 csv 风格表格块。"""
    rows = [r.split(",") for r in block]
    rows = [[c.strip() for c in r] for r in rows]
    header, body = rows[0], rows[1:]
    data = [dict(zip(header, r)) for r in body]
    return {"header": header, "rows": data, "row_count": len(data)}


@register_tool("count_words", "统计字数/词数/行数", "text_document",
               params=[{"name": "text", "type": "str", "description": "待统计文本"},
                       {"name": "language", "type": "str", "description": "语言 (en/zh/auto)",
                        "required": False, "default": "auto"}])
def count_words(text, language="auto"):
    """统计文本的字符数、词数、行数。"""
    try:
        lines = text.splitlines()
        chars = len(text)
        chars_no_space = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        # 中文字符数
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # 英文单词数
        en_words = len(re.findall(r"[A-Za-z]+", text))
        if language == "zh":
            words = cn_chars + en_words
        elif language == "en":
            words = en_words
        else:  # auto
            words = cn_chars + en_words
        return {
            "status": "success",
            "chars": chars,
            "chars_no_space": chars_no_space,
            "chinese_chars": cn_chars,
            "english_words": en_words,
            "words": words,
            "lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("format_number", "数字格式化（千分位/小数位/科学计数法）", "text_document",
               params=[{"name": "number", "type": "float", "description": "要格式化的数字"},
                       {"name": "style", "type": "str", "description": "格式风格 (thousands/fixed/scientific/percent)",
                        "required": False, "default": "thousands"},
                       {"name": "decimals", "type": "int", "description": "小数位数",
                        "required": False, "default": 2}])
def format_number(number, style="thousands", decimals=2):
    """数字格式化。"""
    try:
        number = float(number)
        if style == "thousands":
            result = f"{number:,.{decimals}f}"
        elif style == "fixed":
            result = f"{number:.{decimals}f}"
        elif style == "scientific":
            result = f"{number:.{decimals}e}"
        elif style == "percent":
            result = f"{number * 100:.{decimals}f}%"
        elif style == "auto":
            if abs(number) >= 1e6 or (0 < abs(number) < 1e-3):
                result = f"{number:.{decimals}e}"
            else:
                result = f"{number:,.{decimals}f}"
        else:
            return {"status": "error", "reason": f"不支持的格式风格: {style}"}
        return {"status": "success", "input": number, "result": result, "style": style}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("convert_case", "大小写/命名风格转换", "text_document",
               params=[{"name": "text", "type": "str", "description": "待转换文本"},
                       {"name": "to", "type": "str", "description": "目标风格 (upper/lower/title/camel/pascal/snake/kebab)"}])
def convert_case(text, to):
    """大小写与命名风格转换。"""
    try:
        if to == "upper":
            return {"status": "success", "result": text.upper()}
        if to == "lower":
            return {"status": "success", "result": text.lower()}
        if to == "title":
            return {"status": "success", "result": text.title()}
        if to == "snake":
            return {"status": "success", "result": _to_snake(text)}
        if to == "kebab":
            return {"status": "success", "result": _to_snake(text).replace("_", "-")}
        if to == "camel":
            return {"status": "success", "result": _to_camel(text)}
        if to == "pascal":
            camel = _to_camel(text)
            return {"status": "success", "result": camel[:1].upper() + camel[1:] if camel else ""}
        return {"status": "error", "reason": f"不支持的转换目标: {to}"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def _to_snake(text):
    """转换为 snake_case。"""
    s = re.sub(r"[\-\s]+", "_", text.strip())
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _to_camel(text):
    """转换为 camelCase。"""
    parts = _to_snake(text).split("_")
    if not parts:
        return ""
    return parts[0].lower() + "".join(p.title() for p in parts[1:])


@register_tool("extract_keywords", "简单关键词提取（基于词频）", "text_document",
               params=[{"name": "text", "type": "str", "description": "待提取文本"},
                       {"name": "top_n", "type": "int", "description": "返回前 N 个关键词",
                        "required": False, "default": 10},
                       {"name": "min_length", "type": "int", "description": "关键词最小长度",
                        "required": False, "default": 2},
                       {"name": "language", "type": "str", "description": "语言 (en/zh/auto)",
                        "required": False, "default": "auto"}])
def extract_keywords(text, top_n=10, min_length=2, language="auto"):
    """基于词频提取关键词。"""
    try:
        tokens = []
        # 英文 token
        en_tokens = [w.lower() for w in re.findall(r"[A-Za-z]{%d,}" % min_length, text)]
        # 中文 token（按 2-3 字符滑窗）
        cn_text = re.findall(r"[\u4e00-\u9fff]+", text)
        cn_tokens = []
        for seg in cn_text:
            for size in (2, 3):
                for j in range(len(seg) - size + 1):
                    cn_tokens.append(seg[j:j + size])

        if language == "en":
            tokens = en_tokens
        elif language == "zh":
            tokens = cn_tokens
        else:
            tokens = en_tokens + cn_tokens

        # 过滤停用词
        tokens = [t for t in tokens if t not in _STOPWORDS]
        if not tokens:
            return {"status": "success", "keywords": [], "count": 0}
        counter = Counter(tokens)
        keywords = [{"word": w, "count": c} for w, c in counter.most_common(top_n)]
        return {"status": "success", "keywords": keywords,
                "count": len(keywords), "total_unique": len(counter)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}
