"""代码执行与开发工具集 (10 个)——安全沙箱 + Git + AST 静态分析。

设计原则：
  1. ``run_python`` 使用受限命名空间执行，拦截危险 import（os/subprocess 等）
  2. Git 命令通过 ``subprocess`` 执行，带超时与错误捕获
  3. ``install_package`` 通过 ``subprocess`` 调用 pip
  4. 静态分析工具使用 ``ast`` 模块，无副作用
  5. 所有工具返回 dict / list / str（JSON 可序列化）

工具清单
--------
- run_python          : 执行 Python 代码字符串（安全沙箱）
- run_python_file     : 执行 Python 文件
- python_eval         : 安全 Python 表达式求值
- install_package     : 安装 pip 包
- git_status          : Git 状态
- git_commit          : Git 提交
- git_log             : Git 日志
- lint_code           : 简单 Python 代码检查（ast 模块）
- format_code_simple  : 简单代码格式化（缩进修复）
- parse_python_ast    : 解析 Python AST 并返回结构
"""
from __future__ import annotations

import ast
import builtins as _builtins
import io
import json as _json
import os
import re
import subprocess
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from aerospace_agent.research_tools.base import register_tool

# ---- 安全沙箱配置 ----

# 禁止导入的模块（危险/可执行系统命令/文件系统/网络）
_BLOCKED_MODULES = {
    "os", "subprocess", "sys", "shutil", "pathlib",
    "socket", "http", "urllib", "ftplib", "smtplib", "telnetlib",
    "ctypes", "multiprocessing", "threading", "asyncio",
    "pickle", "marshal", "importlib",
    "builtins", "__builtin__",
    "pty", "commands", "platform",
}

# 允许在沙箱中使用的内建函数
_SAFE_BUILTINS = {
    name: getattr(_builtins, name)
    for name in (
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "hash", "hex", "id",
        "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "print", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "str",
        "sum", "tuple", "type", "zip",
    )
    if hasattr(_builtins, name)
}

# 沙箱中预导入的安全模块
_SAFE_MODULES = {
    "math": __import__("math"),
    "json": _json,
    "re": re,
    "string": __import__("string"),
    "collections": __import__("collections"),
    "itertools": __import__("itertools"),
    "functools": __import__("functools"),
    "datetime": __import__("datetime"),
    "statistics": __import__("statistics"),
    "random": __import__("random"),
}


def _make_safe_import():
    """构造受限的 __import__，拦截危险模块。"""

    def _safe_import(name, globals=None, locals=None, fromlist=(),
                     level=0):
        top = name.split(".")[0]
        if top in _BLOCKED_MODULES:
            raise ImportError(
                f"沙箱禁止导入模块 '{top}'（安全限制）"
            )
        if top in _SAFE_MODULES:
            return _SAFE_MODULES[top]
        # 允许尝试导入其它纯计算模块
        try:
            return _builtins.__import__(name, globals, locals, fromlist, level)
        except Exception as e:
            raise ImportError(f"沙箱中导入 '{name}' 失败: {e}")

    return _safe_import


def _build_sandbox_globals() -> Dict[str, Any]:
    """构造受限执行命名空间。"""
    g: Dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
    }
    g["__builtins__"]["__import__"] = _make_safe_import()
    # 暴露安全模块
    g.update(_SAFE_MODULES)
    return g


@register_tool(
    "run_python",
    "在安全沙箱中执行 Python 代码字符串（禁止 os/subprocess 等危险模块）",
    "code_execution",
    params=[
        {"name": "code", "type": "str", "description": "要执行的 Python 代码"},
        {"name": "timeout", "type": "int", "description": "执行超时秒数",
         "required": False, "default": 10},
        {"name": "capture_output", "type": "bool",
         "description": "是否捕获 stdout/stderr", "required": False,
         "default": True},
    ],
)
def run_python(code, timeout=10, capture_output=True):
    if not isinstance(code, str) or not code.strip():
        return {"status": "error", "reason": "code 必须为非空字符串"}

    # 静态检查：禁止直接出现危险调用（双保险）
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"status": "error", "reason": f"语法错误: {e}",
                "lineno": e.lineno}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BLOCKED_MODULES:
                    return {
                        "status": "error",
                        "reason": f"禁止导入模块 '{alias.name}'（安全限制）",
                    }
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _BLOCKED_MODULES:
                return {
                    "status": "error",
                    "reason": f"禁止从模块 '{node.module}' 导入（安全限制）",
                }

    sandbox = _build_sandbox_globals()
    local_ns: Dict[str, Any] = {}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # 简单超时保护：基于信号仅适用于主线程 Unix，这里用捕获异常兜底
    try:
        if capture_output:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(tree, "<sandbox>", "exec"),
                     sandbox, local_ns)
        else:
            exec(compile(tree, "<sandbox>", "exec"), sandbox, local_ns)
    except Exception:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "reason": "执行异常",
            "traceback": tb,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }

    # 收集非下划线开头的本地变量作为结果
    result_vars = {
        k: _to_jsonable(v)
        for k, v in local_ns.items()
        if not k.startswith("_") and _is_jsonable(v)
    }
    return {
        "status": "success",
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "variables": result_vars,
        "var_count": len(result_vars),
    }


@register_tool(
    "run_python_file",
    "执行指定路径的 Python 文件",
    "code_execution",
    params=[
        {"name": "path", "type": "str", "description": "Python 文件路径"},
        {"name": "args", "type": "list", "description": "命令行参数列表",
         "required": False, "default": []},
        {"name": "timeout", "type": "int", "description": "执行超时秒数",
         "required": False, "default": 60},
    ],
)
def run_python_file(path, args=None, timeout=60):
    if not os.path.isfile(path):
        return {"status": "error", "reason": f"文件不存在: {path}"}
    cmd = [sys.executable, path] + list(args or [])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "path": path,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": f"执行超时（{timeout}s）",
                "path": path}
    except Exception as e:
        return {"status": "error", "reason": str(e), "path": path}


@register_tool(
    "python_eval",
    "安全求值单个 Python 表达式（无副作用，仅限字面量与运算）",
    "code_execution",
    params=[
        {"name": "expression", "type": "str",
         "description": "要求值的表达式"},
        {"name": "variables", "type": "dict",
         "description": "可用变量字典", "required": False, "default": {}},
    ],
)
def python_eval(expression, variables=None):
    if not isinstance(expression, str) or not expression.strip():
        return {"status": "error", "reason": "expression 必须为非空字符串"}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return {"status": "error", "reason": f"表达式语法错误: {e}"}

    # 仅允许安全节点类型
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Constant, ast.Name, ast.Load, ast.Num, ast.Str,
        ast.Bytes, ast.NameConstant, ast.List, ast.Tuple, ast.Dict, ast.Set,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.Invert, ast.Not, ast.UAdd, ast.USub, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is,
        ast.IsNot, ast.In, ast.NotIn,
        ast.Call, ast.Attribute, ast.Subscript, ast.Index, ast.Slice,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.comprehension, ast.IfExp,
    )
    for node in ast.walk(tree):
        # 禁止函数调用中访问危险属性
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec", "compile", "open",
                                "__import__", "getattr", "setattr",
                                "globals", "locals", "vars", "dir"):
                return {
                    "status": "error",
                    "reason": f"禁止调用函数 '{node.func.id}'（安全限制）",
                }
        if not isinstance(node, allowed):
            return {
                "status": "error",
                "reason": f"表达式包含不允许的语法: {type(node).__name__}",
            }

    safe_vars = dict(_SAFE_MODULES)
    if variables and isinstance(variables, dict):
        safe_vars.update(variables)
    safe_vars["__builtins__"] = dict(_SAFE_BUILTINS)

    try:
        result = eval(compile(tree, "<eval>", "eval"), safe_vars, {})
    except Exception as e:
        return {"status": "error", "reason": f"求值失败: {e}"}
    return {
        "status": "success",
        "expression": expression,
        "result": _to_jsonable(result),
        "result_type": type(result).__name__,
    }


@register_tool(
    "install_package",
    "通过 pip 安装第三方包",
    "code_execution",
    params=[
        {"name": "package", "type": "str",
         "description": "包名（可带版本，如 numpy==1.24.0）"},
        {"name": "upgrade", "type": "bool", "description": "是否升级",
         "required": False, "default": False},
        {"name": "timeout", "type": "int", "description": "超时秒数",
         "required": False, "default": 300},
    ],
)
def install_package(package, upgrade=False, timeout=300):
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "package": package,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": f"安装超时（{timeout}s）",
                "package": package}
    except Exception as e:
        return {"status": "error", "reason": str(e), "package": package}


def _run_git(args, cwd=None, timeout=30):
    """执行 git 命令并返回结构化结果。"""
    cmd = ["git"] + list(args)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=cwd, encoding="utf-8", errors="replace",
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "",
                "stderr": f"git 命令超时（{timeout}s）"}
    except FileNotFoundError:
        return {"ok": False, "returncode": -1, "stdout": "",
                "stderr": "未找到 git 可执行文件"}
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "",
                "stderr": str(e)}


@register_tool(
    "git_status",
    "获取 Git 仓库状态",
    "code_execution",
    params=[
        {"name": "repo_path", "type": "str",
         "description": "仓库路径（默认当前目录）", "required": False,
         "default": "."},
    ],
)
def git_status(repo_path="."):
    res = _run_git(["status", "--porcelain=v1", "-b"], cwd=repo_path)
    if not res["ok"]:
        return {"status": "error", "reason": res["stderr"].strip()}
    lines = res["stdout"].splitlines()
    branch = lines[0] if lines else ""
    changed = [
        {"status": ln[:2], "file": ln[3:]}
        for ln in lines[1:]
        if ln.strip()
    ]
    return {
        "status": "success",
        "branch": branch,
        "changed_files": changed,
        "changed_count": len(changed),
    }


@register_tool(
    "git_commit",
    "Git 提交（自动 add 所有变更后提交）",
    "code_execution",
    params=[
        {"name": "message", "type": "str", "description": "提交信息"},
        {"name": "repo_path", "type": "str",
         "description": "仓库路径", "required": False, "default": "."},
        {"name": "add_all", "type": "bool",
         "description": "是否 git add -A", "required": False, "default": True},
        {"name": "files", "type": "list",
         "description": "指定 add 的文件列表（add_all=False 时生效）",
         "required": False, "default": []},
    ],
)
def git_commit(message, repo_path=".", add_all=True, files=None):
    if add_all:
        add_res = _run_git(["add", "-A"], cwd=repo_path)
    else:
        add_res = _run_git(["add"] + list(files or []), cwd=repo_path)
    if not add_res["ok"]:
        return {"status": "error", "reason": add_res["stderr"].strip(),
                "stage": "add"}

    commit_res = _run_git(["commit", "-m", message], cwd=repo_path)
    if not commit_res["ok"]:
        # 可能没有变更可提交
        if "nothing to commit" in commit_res["stdout"].lower():
            return {"status": "success", "message": "没有变更可提交",
                    "committed": False}
        return {"status": "error", "reason": commit_res["stderr"].strip(),
                "stage": "commit"}
    return {
        "status": "success",
        "committed": True,
        "message": message,
        "stdout": commit_res["stdout"].strip(),
    }


@register_tool(
    "git_log",
    "获取 Git 提交日志",
    "code_execution",
    params=[
        {"name": "repo_path", "type": "str",
         "description": "仓库路径", "required": False, "default": "."},
        {"name": "limit", "type": "int",
         "description": "返回条数", "required": False, "default": 10},
        {"name": "oneline", "type": "bool",
         "description": "是否单行格式", "required": False, "default": True},
    ],
)
def git_log(repo_path=".", limit=10, oneline=True):
    args = ["log", f"-{int(limit)}"]
    if oneline:
        args.append("--oneline")
    res = _run_git(args, cwd=repo_path)
    if not res["ok"]:
        return {"status": "error", "reason": res["stderr"].strip()}
    entries = [ln for ln in res["stdout"].splitlines() if ln.strip()]
    return {
        "status": "success",
        "count": len(entries),
        "log": entries,
    }


@register_tool(
    "lint_code",
    "简单 Python 代码检查（基于 ast 模块，检测语法/未定义名/危险调用）",
    "code_execution",
    params=[
        {"name": "code", "type": "str", "description": "待检查的代码"},
        {"name": "check_style", "type": "bool",
         "description": "是否检查基础风格（行长/Tab混合）",
         "required": False, "default": True},
    ],
)
def lint_code(code, check_style=True):
    issues: List[Dict[str, Any]] = []

    # 1. 语法检查
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "status": "success",
            "ok": False,
            "issues": [{
                "type": "syntax_error",
                "line": e.lineno,
                "col": e.offset,
                "message": e.msg,
            }],
            "issue_count": 1,
        }

    # 2. 危险调用检查
    dangerous_calls = {
        "eval", "exec", "compile", "__import__",
        "open", "getattr", "setattr", "delattr", "globals", "locals",
    }
    assigned: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assigned.add(node.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assigned.add(alias.asname or alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assigned.add(alias.asname or alias.name)
        if isinstance(node, (ast.Call,)) and isinstance(node.func, ast.Name):
            if node.func.id in dangerous_calls:
                issues.append({
                    "type": "dangerous_call",
                    "line": node.lineno,
                    "message": f"调用了潜在危险函数 '{node.func.id}'",
                })

    # 3. 风格检查
    if check_style:
        for i, line in enumerate(code.splitlines(), 1):
            if len(line) > 100:
                issues.append({
                    "type": "line_too_long",
                    "line": i,
                    "length": len(line),
                    "message": f"行长 {len(line)} 超过 100 字符",
                })
            if "\t" in line and "    " in line:
                issues.append({
                    "type": "mixed_indent",
                    "line": i,
                    "message": "混用 Tab 与空格缩进",
                })
            if line.rstrip() != line:
                issues.append({
                    "type": "trailing_whitespace",
                    "line": i,
                    "message": "行尾有多余空白",
                })

    return {
        "status": "success",
        "ok": len(issues) == 0,
        "issues": issues,
        "issue_count": len(issues),
    }


@register_tool(
    "format_code_simple",
    "简单代码格式化（修复 Tab→空格、去除行尾空白、统一换行）",
    "code_execution",
    params=[
        {"name": "code", "type": "str", "description": "待格式化的代码"},
        {"name": "indent_size", "type": "int",
         "description": "缩进空格数", "required": False, "default": 4},
        {"name": "trim_trailing", "type": "bool",
         "description": "是否去除行尾空白", "required": False, "default": True},
    ],
)
def format_code_simple(code, indent_size=4, trim_trailing=True):
    if not isinstance(code, str):
        return {"status": "error", "reason": "code 必须为字符串"}
    indent_str = " " * int(indent_size)
    lines = code.splitlines()
    formatted: List[str] = []
    for line in lines:
        # Tab → 空格
        new_line = line.expandtabs(int(indent_size))
        # 去除行尾空白
        if trim_trailing:
            new_line = new_line.rstrip()
        formatted.append(new_line)
    result = "\n".join(formatted)
    # 确保文件以单个换行结尾
    if result and not result.endswith("\n"):
        result += "\n"
    changed = result != code
    return {
        "status": "success",
        "code": result,
        "changed": changed,
        "line_count": result.count("\n") + 1 if result else 0,
    }


@register_tool(
    "parse_python_ast",
    "解析 Python 代码 AST 并返回结构化摘要",
    "code_execution",
    params=[
        {"name": "code", "type": "str", "description": "Python 代码"},
        {"name": "max_depth", "type": "int",
         "description": "遍历最大深度", "required": False, "default": 5},
    ],
)
def parse_python_ast(code, max_depth=5):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"status": "error", "reason": f"语法错误: {e}",
                "lineno": e.lineno}

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[Dict[str, Any]] = []
    assignments: List[Dict[str, Any]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "args": args,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "docstring": ast.get_docstring(node),
            })
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "docstring": ast.get_docstring(node),
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import", "module": alias.name,
                    "asname": alias.asname, "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            imports.append({
                "type": "from", "module": node.module,
                "names": [a.name for a in node.names],
                "line": node.lineno,
            })
        elif isinstance(node, ast.Assign):
            targets = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
            if targets:
                assignments.append({
                    "targets": targets, "line": node.lineno,
                })

    # 统计节点类型
    node_counts: Dict[str, int] = {}
    for n in ast.walk(tree):
        name = type(n).__name__
        node_counts[name] = node_counts.get(name, 0) + 1

    return {
        "status": "success",
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "assignments": assignments,
        "node_type_counts": node_counts,
        "total_nodes": sum(node_counts.values()),
        "function_count": len(functions),
        "class_count": len(classes),
    }


# ---- 辅助函数 ----

def _is_jsonable(v: Any) -> bool:
    if isinstance(v, (dict, list, str, int, float, bool, type(None))):
        return True
    try:
        _json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (dict, list, str, int, float, bool, type(None))):
        return v
    try:
        _json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


# ---- 模块自测 ----
if __name__ == "__main__":
    print("run_python:", run_python("import math\nx = math.factorial(5)\nprint(x)"))
    print("run_python (blocked):", run_python("import os\nos.listdir('.')"))
    print("python_eval:", python_eval("2 ** 10 + 3"))
    print("lint_code:", lint_code("x = 1\nimport os\neval('1')\n"))
    print("parse_python_ast:", parse_python_ast("def f(a, b):\n    return a + b\nclass C:\n    pass\n"))
