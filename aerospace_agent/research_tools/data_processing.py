"""数据处理工具集（15 个原子工具）。

第一性原理：
  1. 数据操作是不可分解的科研流水线原子（解析/过滤/排序/聚合/变换）
  2. 仅依赖标准库（csv/json），YAML 可选（不可用时优雅降级）
  3. 所有输入输出 JSON 可序列化——链式组合无损
  4. 统一错误协议：失败时返回 {"status":"error","reason":"..."}
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List

from aerospace_agent.research_tools.base import register_tool

# YAML 可选导入（不可用时优雅降级）
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _yaml = None
    _HAS_YAML = False

# 安全求值时允许的内置函数白名单
_SAFE_BUILTINS = {
    "str": str, "int": int, "float": float, "bool": bool, "len": len,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "list": list, "dict": dict, "set": set,
    "upper": lambda s: s.upper(), "lower": lambda s: s.lower(),
    "title": lambda s: s.title(), "strip": lambda s: s.strip(),
}


def _eval_func(expr: str):
    """将字符串编译为可调用函数（支持 lambda 表达式或内置函数名）。"""
    expr = expr.strip()
    # 1. 直接是白名单中的函数名
    if expr in _SAFE_BUILTINS:
        return _SAFE_BUILTINS[expr]
    # 2. lambda 表达式
    if expr.startswith("lambda"):
        return eval(expr, {"__builtins__": _SAFE_BUILTINS}, {})
    # 3. 普通表达式（视 x 为单个元素），包装为 lambda x: <expr>
    return eval(f"lambda x: ({expr})", {"__builtins__": _SAFE_BUILTINS}, {})


@register_tool("parse_csv", "解析 CSV 数据为字典列表", "data_processing",
               params=[{"name": "content", "type": "str", "description": "CSV 文本内容"},
                       {"name": "delimiter", "type": "str", "description": "字段分隔符",
                        "required": False, "default": ","},
                       {"name": "has_header", "type": "bool", "description": "首行是否为表头",
                        "required": False, "default": True}])
def parse_csv(content, delimiter=",", has_header=True):
    """解析 CSV 文本为字典列表。"""
    try:
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return {"status": "success", "data": [], "count": 0}
        if has_header:
            header = rows[0]
            data = [dict(zip(header, r)) for r in rows[1:]]
        else:
            data = [{"col_%d" % i: v for i, v in enumerate(r)} for r in rows]
        return {"status": "success", "data": data, "count": len(data)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("write_csv", "将字典列表写入 CSV 文本", "data_processing",
               params=[{"name": "data", "type": "list", "description": "字典列表"},
                       {"name": "delimiter", "type": "str", "description": "字段分隔符",
                        "required": False, "default": ","},
                       {"name": "fields", "type": "list", "description": "指定字段顺序",
                        "required": False, "default": None}])
def write_csv(data, delimiter=",", fields=None):
    """将字典列表序列化为 CSV 文本。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        if not data:
            return {"status": "success", "content": "", "count": 0}
        if fields is None:
            # 保持首次出现的顺序
            seen = []
            for row in data:
                for k in row.keys():
                    if k not in seen:
                        seen.append(k)
            fields = seen
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, delimiter=delimiter,
                                extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        return {"status": "success", "content": buf.getvalue(),
                "count": len(data), "fields": fields}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("parse_json", "解析 JSON 字符串为对象", "data_processing",
               params=[{"name": "content", "type": "str", "description": "JSON 字符串"}])
def parse_json(content):
    """解析 JSON 字符串。"""
    try:
        data = json.loads(content)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("write_json", "将对象序列化为 JSON 字符串", "data_processing",
               params=[{"name": "data", "type": "any", "description": "要序列化的对象"},
                       {"name": "indent", "type": "int", "description": "缩进空格数",
                        "required": False, "default": 2},
                       {"name": "ensure_ascii", "type": "bool", "description": "是否转义非 ASCII",
                        "required": False, "default": False}])
def write_json(data, indent=2, ensure_ascii=False):
    """将对象序列化为 JSON 字符串。"""
    try:
        text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii,
                          default=str)
        return {"status": "success", "content": text, "size": len(text)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("parse_yaml", "解析 YAML 字符串为对象", "data_processing",
               params=[{"name": "content", "type": "str", "description": "YAML 文本内容"}])
def parse_yaml(content):
    """解析 YAML 字符串（依赖 PyYAML，不可用时优雅降级）。"""
    if not _HAS_YAML:
        return {"status": "error", "reason": "PyYAML 未安装，请执行 pip install pyyaml"}
    try:
        data = _yaml.safe_load(content)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("write_yaml", "将对象序列化为 YAML 文本", "data_processing",
               params=[{"name": "data", "type": "any", "description": "要序列化的对象"},
                       {"name": "allow_unicode", "type": "bool", "description": "是否允许 Unicode",
                        "required": False, "default": True}])
def write_yaml(data, allow_unicode=True):
    """将对象序列化为 YAML 文本（依赖 PyYAML，不可用时优雅降级）。"""
    if not _HAS_YAML:
        return {"status": "error", "reason": "PyYAML 未安装，请执行 pip install pyyaml"}
    try:
        text = _yaml.safe_dump(data, allow_unicode=allow_unicode, sort_keys=False)
        return {"status": "success", "content": text, "size": len(text)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("filter_data", "按条件过滤数据列表", "data_processing",
               params=[{"name": "data", "type": "list", "description": "字典列表"},
                       {"name": "condition", "type": "str", "description": "过滤条件：dict 形如 '{\"age\":18}' 精确匹配，或表达式字符串如 \"row['age']>18\""}])
def filter_data(data, condition):
    """按条件过滤数据列表。condition 可为 JSON 字符串(精确匹配)或表达式字符串。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        # 尝试把 condition 当作 JSON dict 解析（精确匹配）
        cond_dict = None
        try:
            parsed = json.loads(condition)
            if isinstance(parsed, dict):
                cond_dict = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        if cond_dict is not None:
            result = [row for row in data
                      if all(row.get(k) == v for k, v in cond_dict.items())]
        else:
            # 表达式模式：row 为当前行
            pred = eval(f"lambda row: ({condition})",
                        {"__builtins__": _SAFE_BUILTINS}, {})
            result = [row for row in data if pred(row)]
        return {"status": "success", "data": result,
                "count": len(result), "total": len(data)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("sort_data", "按键排序数据", "data_processing",
               params=[{"name": "data", "type": "list", "description": "数据列表"},
                       {"name": "key", "type": "str", "description": "排序键（dict 取该字段；list 元素留空按元素本身）",
                        "required": False, "default": ""},
                       {"name": "reverse", "type": "bool", "description": "是否降序",
                        "required": False, "default": False}])
def sort_data(data, key="", reverse=False):
    """按键排序数据列表。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        if key:
            result = sorted(data, key=lambda r: r.get(key, None) if isinstance(r, dict) else r,
                            reverse=reverse)
        else:
            result = sorted(data, key=lambda r: r, reverse=reverse)
        return {"status": "success", "data": result, "count": len(result)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("merge_data", "合并多个数据列表", "data_processing",
               params=[{"name": "lists", "type": "list", "description": "待合并的列表的列表"},
                       {"name": "deduplicate", "type": "bool", "description": "是否去重",
                        "required": False, "default": False}])
def merge_data(lists, deduplicate=False):
    """合并多个数据列表。lists 为列表的列表。"""
    try:
        merged = []
        for lst in lists:
            merged.extend(lst)
        if deduplicate:
            seen, uniq = [], []
            for item in merged:
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    seen.append(key)
                    uniq.append(item)
            merged = uniq
        return {"status": "success", "data": merged, "count": len(merged)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("transform_data", "对数据列表应用变换函数", "data_processing",
               params=[{"name": "data", "type": "list", "description": "数据列表"},
                       {"name": "func", "type": "str", "description": "变换函数：lambda 表达式如 'lambda x: x*2'，或表达式如 'x+1'，或函数名如 'str'"},
                       {"name": "field", "type": "str", "description": "若元素为 dict，作用于该字段（留空则作用于整个元素）",
                        "required": False, "default": ""}])
def transform_data(data, func, field=""):
    """对数据列表应用变换函数。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        fn = _eval_func(func)
        result = []
        for item in data:
            if field and isinstance(item, dict):
                new = dict(item)
                new[field] = fn(item.get(field))
                result.append(new)
            else:
                result.append(fn(item))
        return {"status": "success", "data": result, "count": len(result)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("aggregate_data", "按键分组聚合 (count/sum/avg/min/max)", "data_processing",
               params=[{"name": "data", "type": "list", "description": "字典列表"},
                       {"name": "group_by", "type": "str", "description": "分组字段"},
                       {"name": "agg_field", "type": "str", "description": "聚合数值字段"},
                       {"name": "agg_func", "type": "str", "description": "聚合函数 (count/sum/avg/min/max)",
                        "required": False, "default": "count"}])
def aggregate_data(data, group_by, agg_field="", agg_func="count"):
    """按键分组聚合。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        groups: Dict[Any, list] = {}
        for row in data:
            if not isinstance(row, dict):
                return {"status": "error", "reason": "元素必须为 dict"}
            gk = row.get(group_by)
            groups.setdefault(gk, []).append(row)
        result = []
        for gk, rows in groups.items():
            if agg_func == "count":
                val = len(rows)
            else:
                if not agg_field:
                    return {"status": "error", "reason": "非 count 聚合需指定 agg_field"}
                nums = [r.get(agg_field, 0) for r in rows]
                nums = [n for n in nums if isinstance(n, (int, float))]
                if agg_func == "sum":
                    val = sum(nums)
                elif agg_func == "avg":
                    val = round(sum(nums) / len(nums), 6) if nums else 0
                elif agg_func == "min":
                    val = min(nums) if nums else None
                elif agg_func == "max":
                    val = max(nums) if nums else None
                else:
                    return {"status": "error", "reason": f"不支持的聚合函数: {agg_func}"}
            result.append({group_by: gk, agg_func: val, "count": len(rows)})
        return {"status": "success", "data": result, "groups": len(result)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("pivot_data", "简单透视表", "data_processing",
               params=[{"name": "data", "type": "list", "description": "字典列表"},
                       {"name": "index", "type": "str", "description": "行索引字段"},
                       {"name": "columns", "type": "str", "description": "列字段"},
                       {"name": "values", "type": "str", "description": "值字段"},
                       {"name": "agg_func", "type": "str", "description": "聚合函数 (sum/avg/count/min/max)",
                        "required": False, "default": "sum"}])
def pivot_data(data, index, columns, values, agg_func="sum"):
    """生成简单透视表。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        # 收集所有行索引与列
        row_keys, col_keys = [], []
        bucket: Dict[tuple, list] = {}
        for row in data:
            rk = row.get(index)
            ck = row.get(columns)
            if rk not in row_keys:
                row_keys.append(rk)
            if ck not in col_keys:
                col_keys.append(ck)
            bucket.setdefault((rk, ck), []).append(row.get(values))
        table = []
        for rk in row_keys:
            rec = {index: rk}
            for ck in col_keys:
                nums = [v for v in bucket.get((rk, ck), []) if isinstance(v, (int, float))]
                if agg_func == "sum":
                    rec[str(ck)] = sum(nums)
                elif agg_func == "avg":
                    rec[str(ck)] = round(sum(nums) / len(nums), 6) if nums else 0
                elif agg_func == "count":
                    rec[str(ck)] = len(bucket.get((rk, ck), []))
                elif agg_func == "min":
                    rec[str(ck)] = min(nums) if nums else None
                elif agg_func == "max":
                    rec[str(ck)] = max(nums) if nums else None
                else:
                    return {"status": "error", "reason": f"不支持的聚合函数: {agg_func}"}
            table.append(rec)
        return {"status": "success", "data": table,
                "rows": len(row_keys), "columns": [str(c) for c in col_keys]}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("validate_schema", "验证数据是否符合给定 schema", "data_processing",
               params=[{"name": "data", "type": "list", "description": "字典列表"},
                       {"name": "schema", "type": "dict", "description": "schema：字段名->类型名(str/int/float/bool/list/dict)"},
                       {"name": "required_strict", "type": "bool", "description": "是否要求所有 schema 字段必须存在",
                        "required": False, "default": True}])
def validate_schema(data, schema, required_strict=True):
    """验证数据列表是否符合给定 schema。"""
    try:
        _type_map = {"str": str, "int": int, "float": float, "bool": bool,
                     "list": list, "dict": dict}
        errors = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                errors.append({"row": i, "error": "元素不是 dict"})
                continue
            for field, typename in schema.items():
                if field not in row:
                    if required_strict:
                        errors.append({"row": i, "field": field, "error": "缺少字段"})
                    continue
                expected = _type_map.get(typename)
                if expected is None:
                    errors.append({"row": i, "field": field, "error": f"未知类型: {typename}"})
                    continue
                # int 兼容 bool 排除（bool 是 int 子类）
                val = row[field]
                if expected is int and isinstance(val, bool):
                    errors.append({"row": i, "field": field, "error": f"类型不符，期望 {typename}，实际 bool"})
                elif not isinstance(val, expected):
                    errors.append({"row": i, "field": field,
                                   "error": f"类型不符，期望 {typename}，实际 {type(val).__name__}"})
        return {"status": "success", "valid": len(errors) == 0,
                "errors": errors, "error_count": len(errors),
                "checked": len(data)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("diff_data", "比较两个数据集的差异", "data_processing",
               params=[{"name": "data_a", "type": "list", "description": "数据集 A（字典列表）"},
                       {"name": "data_b", "type": "list", "description": "数据集 B（字典列表）"},
                       {"name": "key", "type": "str", "description": "用于比对的唯一键字段"}])
def diff_data(data_a, data_b, key):
    """比较两个数据集差异，返回新增/删除/修改项。"""
    try:
        map_a = {row.get(key): row for row in data_a if isinstance(row, dict)}
        map_b = {row.get(key): row for row in data_b if isinstance(row, dict)}
        keys_a, keys_b = set(map_a.keys()), set(map_b.keys())
        added = [map_b[k] for k in keys_b - keys_a]
        removed = [map_a[k] for k in keys_a - keys_b]
        modified = []
        for k in keys_a & keys_b:
            sa = json.dumps(map_a[k], sort_keys=True, default=str)
            sb = json.dumps(map_b[k], sort_keys=True, default=str)
            if sa != sb:
                modified.append({"key": k, "from": map_a[k], "to": map_b[k]})
        return {"status": "success", "added": added, "removed": removed,
                "modified": modified,
                "summary": {"added": len(added), "removed": len(removed),
                            "modified": len(modified)}}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("unique_values", "提取唯一值", "data_processing",
               params=[{"name": "data", "type": "list", "description": "数据列表（dict 列表或标量列表）"},
                       {"name": "field", "type": "str", "description": "若元素为 dict，提取该字段（留空则作用于整个元素）",
                        "required": False, "default": ""}])
def unique_values(data, field=""):
    """提取数据列表中的唯一值。"""
    try:
        if not isinstance(data, list):
            return {"status": "error", "reason": "data 必须为列表"}
        seen, uniq = [], []
        for item in data:
            val = item.get(field) if field and isinstance(item, dict) else item
            k = json.dumps(val, sort_keys=True, default=str)
            if k not in seen:
                seen.append(k)
                uniq.append(val)
        return {"status": "success", "values": uniq, "count": len(uniq),
                "total": len(data)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}
