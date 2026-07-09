"""自进化工具集 (5 个)——动态工具创建 / 列举 / 帮助 / 校验 / 管道组合。

这是 Agent 自我进化的核心：当所需工具不存在时，Agent 可用 ``create_tool``
编写并注册新工具；用 ``compose_pipeline`` 将多个工具组合为执行管道。

第一性原理：
  1. 工具是原子操作，可通过组合表达任意工作流
  2. 工具集非封闭——运行时可动态扩展（create_tool）
  3. 组合优于继承——compose_pipeline 让工具 A 的输出成为工具 B 的输入
  4. 自省——list_tools / tool_help 让 Agent 知道"我能做什么"

工具清单
--------
- create_tool      : 动态创建新工具（调用 registry.create_dynamic_tool）
- list_tools       : 列出所有可用工具（按分类）
- tool_help        : 获取工具的详细帮助与 schema
- validate_output  : 验证工具输出是否符合预期 schema
- compose_pipeline : 组合多个工具为执行管道
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from aerospace_agent.research_tools.base import get_registry, register_tool


@register_tool(
    "create_tool",
    "动态创建新工具——当所需工具不存在时自我进化",
    "self_evolution",
    params=[
        {"name": "name", "type": "str",
         "description": "工具名（snake_case）"},
        {"name": "description", "type": "str", "description": "工具描述"},
        {"name": "category", "type": "str", "description": "工具分类"},
        {"name": "code", "type": "str",
         "description": "Python 函数代码，必须包含 def func(...): 或 def <name>(...):"},
        {"name": "params", "type": "list",
         "description": "参数规格列表 [{name,type,description,required,default}]",
         "required": False, "default": []},
    ],
)
def create_tool(name, description, category, code, params=None):
    registry = get_registry()
    return registry.create_dynamic_tool(
        name, description, category, code, params
    )


@register_tool(
    "list_tools",
    "列出所有可用工具（按分类组织）",
    "self_evolution",
    params=[
        {"name": "category", "type": "str",
         "description": "仅列出指定分类（留空列出全部）",
         "required": False, "default": ""},
        {"name": "format", "type": "str",
         "description": "输出格式：summary（紧凑说明）或 detail（含参数）",
         "required": False, "default": "summary"},
    ],
)
def list_tools(category="", format="summary"):
    registry = get_registry()
    fmt = (format or "summary").lower()

    if category:
        categories = {category: registry.list_by_category(category)}
        if not categories[category]:
            return {
                "status": "success",
                "category": category,
                "tools": [],
                "count": 0,
                "message": f"分类 '{category}' 下无工具",
            }
    else:
        categories = registry.categories()

    result: Dict[str, Any] = {
        "status": "success",
        "total": registry.count(),
        "categories": {},
    }
    total_shown = 0
    for cat, names in sorted(categories.items()):
        cat_tools: List[Any] = []
        for n in names:
            tool = registry.get(n)
            if tool is None:
                continue
            if fmt == "detail":
                cat_tools.append(tool.to_json_schema())
            else:
                cat_tools.append(tool.to_schema())
        result["categories"][cat] = {
            "count": len(cat_tools),
            "tools": cat_tools,
        }
        total_shown += len(cat_tools)
    result["shown"] = total_shown
    return result


@register_tool(
    "tool_help",
    "获取单个工具的详细帮助与参数 schema",
    "self_evolution",
    params=[
        {"name": "name", "type": "str", "description": "工具名"},
        {"name": "format", "type": "str",
         "description": "输出格式：text（人类可读）或 json（JSON Schema）",
         "required": False, "default": "text"},
    ],
)
def tool_help(name, format="text"):
    registry = get_registry()
    tool = registry.get(name)
    if tool is None:
        return {
            "status": "error",
            "reason": f"工具 '{name}' 不存在",
            "available": registry.list_all()[:50],
        }
    fmt = (format or "text").lower()
    if fmt == "json":
        return {
            "status": "success",
            "name": name,
            "schema": tool.to_json_schema(),
        }
    # text 格式：人类可读说明
    lines = [
        f"工具名: {tool.name}",
        f"描述: {tool.description}",
        f"分类: {tool.category}",
        f"版本: {tool.version}",
        f"作者: {tool.author}",
        f"标签: {', '.join(tool.tags) if tool.tags else '(无)'}",
        "",
        "参数:",
    ]
    if tool.params:
        for p in tool.params:
            req = "必填" if p.required else "可选"
            default = f"，默认={p.default!r}" if not p.required else ""
            lines.append(
                f"  - {p.name} ({p.type}, {req}): {p.description}{default}"
            )
    else:
        lines.append("  (无参数)")
    lines.append("")
    lines.append("调用示例:")
    if tool.params:
        args = ", ".join(
            f'{p.name}=...' + ("" if p.required else "  # 可选")
            for p in tool.params
        )
        lines.append(f"  {tool.name}({args})")
    else:
        lines.append(f"  {tool.name}()")
    lines.append("")
    lines.append("返回: dict / list / str（JSON 可序列化）")
    return {
        "status": "success",
        "name": name,
        "help": "\n".join(lines),
        "params": [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
                "default": p.default,
            }
            for p in tool.params
        ],
    }


@register_tool(
    "validate_output",
    "验证工具输出是否符合预期 schema（字段存在性与类型检查）",
    "self_evolution",
    params=[
        {"name": "output", "type": "dict", "description": "待验证的输出"},
        {"name": "schema", "type": "dict",
         "description": "期望结构 {字段名: 类型}，类型为 str/int/float/bool/list/dict/any"},
        {"name": "required_fields", "type": "list",
         "description": "必填字段名列表（默认 schema 全部键）",
         "required": False, "default": []},
        {"name": "allow_extra", "type": "bool",
         "description": "是否允许额外字段", "required": False, "default": True},
    ],
)
def validate_output(output, schema, required_fields=None, allow_extra=True):
    if not isinstance(output, dict):
        return {
            "status": "error",
            "valid": False,
            "reason": f"output 应为 dict，实际为 {type(output).__name__}",
        }
    if not isinstance(schema, dict):
        return {
            "status": "error",
            "valid": False,
            "reason": f"schema 应为 dict，实际为 {type(schema).__name__}",
        }

    required = required_fields if required_fields else list(schema.keys())
    errors: List[str] = []
    checked: List[Dict[str, Any]] = []

    _type_map = {
        "str": str, "int": int, "float": (int, float),
        "bool": bool, "list": list, "dict": dict,
        "any": None,
    }

    # 必填字段存在性
    for field in required:
        if field not in output:
            errors.append(f"缺少必填字段 '{field}'")

    # 类型检查
    for field, expected_type in schema.items():
        if field not in output:
            continue
        actual_val = output[field]
        expected_type = (expected_type or "any").lower()
        py_type = _type_map.get(expected_type)
        entry = {
            "field": field,
            "expected": expected_type,
            "actual": type(actual_val).__name__,
            "ok": True,
        }
        if py_type is None:
            # any 类型，跳过检查
            entry["ok"] = True
        else:
            # bool 是 int 的子类，需特殊处理避免 True 被当作 int
            if expected_type == "int" and isinstance(actual_val, bool):
                entry["ok"] = False
                entry["note"] = "bool 不视为 int"
                errors.append(
                    f"字段 '{field}' 类型不符：期望 int，实际 bool"
                )
            elif expected_type == "float" and isinstance(actual_val, bool):
                entry["ok"] = False
                errors.append(
                    f"字段 '{field}' 类型不符：期望 float，实际 bool"
                )
            elif not isinstance(actual_val, py_type):
                entry["ok"] = False
                errors.append(
                    f"字段 '{field}' 类型不符：期望 {expected_type}，"
                    f"实际 {type(actual_val).__name__}"
                )
        checked.append(entry)

    # 额外字段检查
    extra_fields = []
    if not allow_extra:
        for field in output:
            if field not in schema:
                extra_fields.append(field)
                errors.append(f"存在额外字段 '{field}'（不允许）")

    return {
        "status": "success",
        "valid": len(errors) == 0,
        "errors": errors,
        "error_count": len(errors),
        "checked_fields": checked,
        "extra_fields": extra_fields,
        "output_keys": list(output.keys()),
    }


@register_tool(
    "compose_pipeline",
    "组合多个工具为执行管道：依次调用，上一步结果作为下一步参数传入",
    "self_evolution",
    params=[
        {"name": "tools", "type": "list", "description": "工具名列表"},
        {"name": "initial_input", "type": "dict",
         "description": "初始输入参数（第一个工具的 kwargs）"},
        {"name": "pass_as", "type": "str",
         "description": "上一步结果作为下一步哪个参数传入",
         "required": False, "default": "input"},
    ],
)
def compose_pipeline(tools, initial_input, pass_as="input"):
    registry = get_registry()

    if not isinstance(tools, list) or not tools:
        return {"status": "error", "reason": "tools 必须为非空列表"}
    if not isinstance(initial_input, dict):
        return {"status": "error",
                "reason": "initial_input 必须为 dict"}

    steps: List[Dict[str, Any]] = []
    current_input: Dict[str, Any] = dict(initial_input)
    final_result: Any = None

    for idx, tool_name in enumerate(tools):
        tool = registry.get(tool_name)
        if tool is None:
            steps.append({
                "step": idx + 1,
                "tool": tool_name,
                "status": "error",
                "reason": f"工具 '{tool_name}' 不存在",
            })
            return {
                "status": "error",
                "reason": f"管道在第 {idx + 1} 步中断："
                          f"工具 '{tool_name}' 不存在",
                "completed_steps": idx,
                "steps": steps,
            }

        try:
            result = tool(**current_input)
        except Exception as e:
            steps.append({
                "step": idx + 1,
                "tool": tool_name,
                "status": "error",
                "reason": str(e),
                "input": _safe_serialize(current_input),
            })
            return {
                "status": "error",
                "reason": f"管道在第 {idx + 1} 步执行失败: {e}",
                "failed_step": idx + 1,
                "steps": steps,
            }

        final_result = result
        steps.append({
            "step": idx + 1,
            "tool": tool_name,
            "status": result.get("status", "unknown") if isinstance(
                result, dict) else "ok",
            "result": _safe_serialize(result),
        })

        # 上一步结果作为下一步的 pass_as 参数
        current_input = {pass_as: result}

    return {
        "status": "success",
        "tools": tools,
        "step_count": len(steps),
        "steps": steps,
        "final_result": _safe_serialize(final_result),
        "pass_as": pass_as,
    }


def _safe_serialize(obj: Any, max_str_len: int = 5000) -> Any:
    """安全序列化结果，确保 JSON 可序列化并截断过长字符串。"""
    if isinstance(obj, str):
        return obj if len(obj) <= max_str_len else obj[:max_str_len] + "...[truncated]"
    if isinstance(obj, (dict, list)):
        try:
            import json
            json.dumps(obj)
            # 截断过大的容器
            if isinstance(obj, dict):
                return {
                    k: _safe_serialize(v, max_str_len)
                    for k, v in obj.items()
                }
            return [_safe_serialize(v, max_str_len) for v in obj]
        except (TypeError, ValueError, OverflowError):
            return repr(obj)
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    return repr(obj)


# ---- 模块自测 ----
if __name__ == "__main__":
    print("list_tools:", list_tools())
    print("tool_help:", tool_help("list_tools"))
    print("validate_output (ok):", validate_output(
        {"status": "success", "value": 42},
        {"status": "str", "value": "int"},
    ))
    print("validate_output (fail):", validate_output(
        {"status": "success", "value": "not_int"},
        {"status": "str", "value": "int"},
    ))
    print("create_tool:", create_tool(
        name="square_number",
        description="计算一个数的平方",
        category="math_test",
        params=[{"name": "n", "type": "int", "description": "输入数字"}],
        code="def square_number(n):\n    return {'status':'success','result':n*n}",
    ))
    print("compose_pipeline:",
          compose_pipeline(
              tools=["square_number", "square_number"],
              initial_input={"n": 3},
              pass_as="n",
          ))
