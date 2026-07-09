"""ArchAgent -- 架构统筹 Agent。

职责：依赖图治理 - 配置统一 - 架构红线 - 决策账本

第一性原理
----------
  1. 依赖图必须有向无环 (DAG) -- 循环依赖是架构腐化的第一信号
  2. 配置必须统一入口 -- 散落的环境变量导致配置漂移
  3. 架构红线不可逾越 -- sys.path hack、bare except、上帝类是技术债的红旗
  4. 决策可追溯 -- 所有架构检查结果结构化记录，供编排器决策

检测能力
--------
  - check_dag()                    : 依赖图环检测 (三色标记 DFS)
  - audit_imports()                : import 语句正确性审计
  - audit_god_class()              : 上帝类检测 (方法数/行数超标)
  - unify_config()                 : 配置统一方案生成 (config.yaml 模板)
  - detect_dead_code()             : 死代码检测 (未被引用的 mcp_tools 模块)
  - review_version_consistency()   : 版本号一致性检查
  - enforce_architecture_redlines(): 架构红线综合检查 (聚合以上所有)

设计约束
--------
  - 只做检测和分析，不修改任何文件
  - 所有方法返回 AgentResult
  - 使用 self.log / self.metrics / self._time_operation()
"""
from __future__ import annotations

import ast
import collections
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

from .base import AgentBase, AgentResult, AgentRole


class ArchAgent(AgentBase):
    """架构统筹 Agent -- 守护依赖图 DAG、配置统一、架构红线。

    继承 ``AgentBase``，role = ``AgentRole.ARCH``。

    用法::

        agent = ArchAgent("d:/Project/aerospace-agent")
        result = agent.check_dag()          # 检测循环依赖
        result = agent.audit_imports()      # 审计 import 正确性
        result = agent.enforce_architecture_redlines()  # 综合红线检查
    """

    role = AgentRole.ARCH

    # ------------------------------------------------------------------
    # 常量定义
    # ------------------------------------------------------------------

    #: 内部顶层包名集合 -- 用于判定 import 是否为内部依赖
    _INTERNAL_GROUPS: Set[str] = {
        "core", "rag", "agents", "mcp", "mcp_tools", "physics",
        "prompts", "skills", "reporting", "utils", "workflows",
    }

    #: 环境变量按功能分组的匹配关键词 (大写匹配)
    _CONFIG_GROUPS: Dict[str, List[str]] = {
        "LLM": [
            "LLM", "MODEL", "OPENAI", "API_KEY", "QWEN",
            "OLLAMA", "VLLM", "TOKEN",
        ],
        "RAG": [
            "RAG", "RETRIEVER", "EMBED", "INDEX", "SEARCH",
            "CSTCLOUD",
        ],
        "Engine": [
            "ENGINE", "GMAT", "OREKIT", "BASILISK", "SPICE",
            "STK", "ASTROPY", "POLIASTRO", "KERNEL",
        ],
        "Safety": [
            "SAFETY", "SANDBOX", "LICENSE", "PATH_POLICY",
            "SECURITY", "WORKSPACE",
        ],
        "Data": [
            "DATA", "KB", "KNOWLEDGE", "VECTOR", "MEMORY",
            "PAPER", "REGISTRY",
        ],
        "Observability": [
            "LOG", "METRIC", "TRACE", "MONITOR",
        ],
    }

    #: 上帝类阈值
    _MAX_METHODS = 20
    _MAX_LINES = 500

    #: 严重度排序 (数值越小越严重)
    _SEVERITY_ORDER: Dict[str, int] = {
        "critical": 0, "high": 1, "medium": 2, "low": 3,
    }

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, project_root: str = "."):
        super().__init__(project_root)
        # 规范化路径分隔符 (Windows 兼容)
        self.project_root = os.path.normpath(self.project_root)
        self.package_root = os.path.normpath(self.package_root)

    # ==================================================================
    # 公开方法
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. check_dag -- 依赖图环检测
    # ------------------------------------------------------------------

    def check_dag(self) -> AgentResult:
        """检测循环依赖 -- 扫描所有 .py 文件的 import 语句，构建依赖图并检测环。

        - 用 ``ast.walk`` 扫描 ``aerospace_agent/`` 下所有 .py 文件
        - 提取 import 语句，解析为顶层包级依赖 (如 core -> rag)
        - 检测 ``sys.path.insert`` + ``from core.xxx import`` hack 模式 (标记违规)
        - 构建邻接表，用三色标记 DFS 检测环

        Returns:
            AgentResult, data 含:
                - total_modules: 依赖图中模块 (顶层包) 数
                - edges: 依赖边数
                - cycles: 环列表 [{"path": [...], "type": "circular"}]
                - violations: 违规列表 [{"file", "line", "type", "detail"}]
        """
        start = time.perf_counter()
        with self._time_operation("check_dag"):
            py_files = self._iter_py_files()
            adj: Dict[str, Set[str]] = collections.defaultdict(set)
            violations: List[Dict] = []

            for filepath in py_files:
                rel = os.path.relpath(filepath, self.package_root)
                rel = rel.replace("\\", "/")
                source_group = self._module_group(rel)
                if not source_group:
                    continue

                tree = self._read_ast(filepath)
                if tree is None:
                    continue

                for node in ast.walk(tree):
                    # 检测 sys.path.insert / sys.path.append hack
                    if isinstance(node, ast.Call) and self._is_sys_path_call(node):
                        attr = self._get_call_attr(node)
                        violations.append({
                            "file": rel,
                            "line": node.lineno,
                            "type": "sys_path_hack",
                            "detail": f"sys.path.{attr}() -- "
                                      f"使用 sys.path 修改是绝对违规",
                        })

                    # 提取 import 依赖
                    if isinstance(node, ast.ImportFrom):
                        target = self._resolve_import_from(node, source_group)
                        if target and target != source_group:
                            adj[source_group].add(target)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            target = self._resolve_import(alias.name)
                            if target and target != source_group:
                                adj[source_group].add(target)

            # 确保所有已知内部包都在图中 (即使无边)
            for group in self._INTERNAL_GROUPS:
                if group not in adj:
                    adj[group] = set()

            # 三色标记 DFS 检测环
            cycles = self._detect_cycles_dfs(adj)

            # 构建 issues
            issues: List[Dict] = []
            for v in violations:
                issues.append({
                    "file": v["file"],
                    "line": v["line"],
                    "type": v["type"],
                    "severity": "critical",
                    "detail": v["detail"],
                })
            for c in cycles:
                issues.append({
                    "type": "circular_dependency",
                    "severity": "critical",
                    "path": c["path"],
                    "detail": " -> ".join(c["path"]),
                })

            edge_count = sum(len(targets) for targets in adj.values())

            data = {
                "total_modules": len(adj),
                "edges": edge_count,
                "cycles": cycles,
                "violations": violations,
            }

            self.log.info("check_dag_done", data={
                "modules": len(adj),
                "edges": edge_count,
                "cycles": len(cycles),
                "violations": len(violations),
            })
            self.metrics.gauge("arch.dag.modules", len(adj))
            self.metrics.gauge("arch.dag.edges", edge_count)
            self.metrics.gauge("arch.dag.cycles", len(cycles))
            self.metrics.gauge("arch.dag.sys_path_hacks", len(violations))

            duration = time.perf_counter() - start
            return self._make_result(
                "check_dag", success=True, data=data,
                issues=issues, duration=duration,
            )

    # ------------------------------------------------------------------
    # 2. audit_imports -- import 语句正确性审计
    # ------------------------------------------------------------------

    def audit_imports(self) -> AgentResult:
        """审计所有 import 语句的正确性。

        检测项:
            a) sys.path.insert / append hack (绝对违规, critical)
            b) from core.xxx import (应为 from ..core.xxx, high)
            c) from rag.xxx import (应为 from ..rag.xxx, high)
            d) bare except: (应为 except Exception:, medium)
            e) except Exception: pass (静默吞异常, medium)

        Returns:
            AgentResult, data 含:
                - total_files: 扫描文件数
                - violations: {"sys_path_hack": N, "wrong_relative_import": N,
                               "bare_except": N, "silent_pass": N}
            issues 列表, 每个含 file/line/type/severity/detail
        """
        start = time.perf_counter()
        with self._time_operation("audit_imports"):
            py_files = self._iter_py_files()
            issues: List[Dict] = []
            counts = {
                "sys_path_hack": 0,
                "wrong_relative_import": 0,
                "bare_except": 0,
                "silent_pass": 0,
            }

            for filepath in py_files:
                rel = os.path.relpath(filepath, self.package_root)
                rel = rel.replace("\\", "/")
                tree = self._read_ast(filepath)
                if tree is None:
                    continue

                for node in ast.walk(tree):
                    # a) sys.path.insert / append hack
                    if isinstance(node, ast.Call) and self._is_sys_path_call(node):
                        attr = self._get_call_attr(node)
                        counts["sys_path_hack"] += 1
                        issues.append({
                            "file": rel,
                            "line": node.lineno,
                            "type": "sys_path_hack",
                            "severity": "critical",
                            "detail": f"sys.path.{attr}() -- 绝对违规",
                        })

                    # b/c) 错误的绝对 import (应为相对 import)
                    if (isinstance(node, ast.ImportFrom)
                            and node.level == 0
                            and node.module
                            and self._is_bare_internal_import(node.module)):
                        counts["wrong_relative_import"] += 1
                        issues.append({
                            "file": rel,
                            "line": node.lineno,
                            "type": "wrong_relative_import",
                            "severity": "high",
                            "detail": (f"from {node.module} import -- "
                                       f"应改为相对 import "
                                       f"(from ..{node.module})"),
                        })

                    # d) bare except:
                    if (isinstance(node, ast.ExceptHandler)
                            and node.type is None):
                        counts["bare_except"] += 1
                        issues.append({
                            "file": rel,
                            "line": node.lineno,
                            "type": "bare_except",
                            "severity": "medium",
                            "detail": "bare except: -- 应为 except Exception:",
                        })

                    # e) except Exception: pass (静默吞异常)
                    if (isinstance(node, ast.ExceptHandler)
                            and isinstance(node.type, ast.Name)
                            and node.type.id == "Exception"
                            and len(node.body) == 1
                            and isinstance(node.body[0], ast.Pass)):
                        counts["silent_pass"] += 1
                        issues.append({
                            "file": rel,
                            "line": node.lineno,
                            "type": "silent_pass",
                            "severity": "medium",
                            "detail": "except Exception: pass -- 静默吞异常",
                        })

            data = {
                "total_files": len(py_files),
                "violations": counts,
            }

            total_violations = sum(counts.values())
            self.log.info("audit_imports_done", data={
                "files": len(py_files),
                "violations": total_violations,
                "breakdown": counts,
            })
            for k, v in counts.items():
                self.metrics.gauge(f"arch.imports.{k}", v)

            duration = time.perf_counter() - start
            return self._make_result(
                "audit_imports", success=True, data=data,
                issues=issues, duration=duration,
            )

    # ------------------------------------------------------------------
    # 3. audit_god_class -- 上帝类检测
    # ------------------------------------------------------------------

    def audit_god_class(self) -> AgentResult:
        """检测上帝类 -- 方法数/行数过多的类。

        - 扫描 ``aerospace_agent/core/`` 下所有 .py 文件中的类定义
        - 统计每个类的方法数、总行数、属性数
        - 阈值: 方法数 > 20 或 行数 > 500 -> 标记为上帝类

        Returns:
            AgentResult, data 含:
                - classes: [{"name", "methods", "lines", "attributes",
                             "is_god_class", "file"}]
            issues 列表
        """
        start = time.perf_counter()
        with self._time_operation("audit_god_class"):
            classes_info: List[Dict] = []
            issues: List[Dict] = []

            # 扫描 core/ 目录下所有 .py 文件
            core_dir = os.path.join(self.package_root, "core")
            core_files: List[str] = []
            if os.path.isdir(core_dir):
                for f in sorted(os.listdir(core_dir)):
                    if f.endswith(".py"):
                        core_files.append(os.path.join(core_dir, f))

            for filepath in core_files:
                rel = os.path.relpath(filepath, self.package_root)
                rel = rel.replace("\\", "/")
                tree = self._read_ast(filepath)
                if tree is None:
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef):
                        continue

                    methods = sum(
                        1 for child in node.body
                        if isinstance(child, (ast.FunctionDef,
                                             ast.AsyncFunctionDef))
                    )
                    end_line = getattr(node, "end_lineno", None) or node.lineno
                    lines = end_line - node.lineno + 1
                    attrs = self._count_class_attributes(node)
                    is_god = (methods > self._MAX_METHODS
                              or lines > self._MAX_LINES)

                    info = {
                        "name": node.name,
                        "methods": methods,
                        "lines": lines,
                        "attributes": attrs,
                        "is_god_class": is_god,
                        "file": rel,
                    }
                    classes_info.append(info)

                    if is_god:
                        reasons: List[str] = []
                        if methods > self._MAX_METHODS:
                            reasons.append(
                                f"methods={methods}>{self._MAX_METHODS}")
                        if lines > self._MAX_LINES:
                            reasons.append(
                                f"lines={lines}>{self._MAX_LINES}")
                        issues.append({
                            "file": rel,
                            "line": node.lineno,
                            "type": "god_class",
                            "severity": "high",
                            "detail": (f"{node.name}: "
                                       f"{', '.join(reasons)}"),
                        })

            data = {"classes": classes_info}

            god_count = sum(1 for c in classes_info if c["is_god_class"])
            self.log.info("audit_god_class_done", data={
                "classes": len(classes_info),
                "god_classes": god_count,
            })
            self.metrics.gauge("arch.god_classes.total", len(classes_info))
            self.metrics.gauge("arch.god_classes.flagged", god_count)

            duration = time.perf_counter() - start
            return self._make_result(
                "audit_god_class", success=True, data=data,
                issues=issues, duration=duration,
            )

    # ------------------------------------------------------------------
    # 4. unify_config -- 配置统一方案生成
    # ------------------------------------------------------------------

    def unify_config(self) -> AgentResult:
        """统一配置 -- 扫描散落的环境变量，生成统一配置方案。

        - 扫描所有 .py 文件中的 ``os.environ.get`` / ``os.getenv`` 调用
        - 收集变量名、默认值、所在文件、行号
        - 按功能分组: LLM / Data / Engine / Safety / RAG / Observability
        - 生成 config.yaml 模板内容 (不写文件，只返回内容)

        Returns:
            AgentResult, data 含:
                - total_env_vars: 去重后环境变量数
                - groups: {"LLM": [...], "Data": [...], ...}
                - config_yaml_template: YAML 模板字符串
        """
        start = time.perf_counter()
        with self._time_operation("unify_config"):
            py_files = self._iter_py_files()
            env_vars: List[Dict] = []

            for filepath in py_files:
                rel = os.path.relpath(filepath, self.package_root)
                rel = rel.replace("\\", "/")
                tree = self._read_ast(filepath)
                if tree is None:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        var_info = self._extract_env_var(node, rel)
                        if var_info:
                            env_vars.append(var_info)

            # 按变量名去重 (保留首次出现)
            seen_names: Set[str] = set()
            unique_vars: List[Dict] = []
            for v in env_vars:
                if v["name"] not in seen_names:
                    seen_names.add(v["name"])
                    unique_vars.append(v)

            # 按功能分组
            groups: Dict[str, List[Dict]] = {
                g: [] for g in self._CONFIG_GROUPS
            }
            groups["Other"] = []
            for v in unique_vars:
                group = self._classify_config_group(v["name"])
                groups[group].append(v)

            # 生成 config.yaml 模板
            config_yaml = self._generate_config_yaml(unique_vars, groups)

            # 只返回非空分组
            non_empty_groups = {
                g: members for g, members in groups.items() if members
            }

            data = {
                "total_env_vars": len(unique_vars),
                "groups": non_empty_groups,
                "config_yaml_template": config_yaml,
            }

            self.log.info("unify_config_done", data={
                "env_vars": len(unique_vars),
                "groups": list(non_empty_groups.keys()),
            })
            self.metrics.gauge("arch.config.env_vars", len(unique_vars))

            duration = time.perf_counter() - start
            return self._make_result(
                "unify_config", success=True, data=data,
                duration=duration,
            )

    # ------------------------------------------------------------------
    # 5. detect_dead_code -- 死代码检测
    # ------------------------------------------------------------------

    def detect_dead_code(self) -> AgentResult:
        """检测死代码 -- 未被引用的 mcp_tools 模块。

        - 扫描 ``aerospace_agent/mcp_tools/`` 下的工具文件
        - 检查它们是否在 ``agent.py`` 的 ``_load_mcp_tools`` 中被加载
        - 检查它们是否在 ``mcp_tools/__init__.py`` 中被导出
        - 标记未被引用的文件为死代码

        Returns:
            AgentResult, data 含:
                - dead_files: [{"file", "module"}]
                - alive_files: [{"file", "module", "referenced_in"}]
                - total_scanned: N
        """
        start = time.perf_counter()
        with self._time_operation("detect_dead_code"):
            mcp_tools_dir = os.path.join(self.package_root, "mcp_tools")

            # 收集 mcp_tools/ 下所有 .py 文件 (排除 __init__.py)
            tool_files: Dict[str, str] = {}  # module_name -> filename
            if os.path.isdir(mcp_tools_dir):
                for f in sorted(os.listdir(mcp_tools_dir)):
                    if f.endswith(".py") and f != "__init__.py":
                        mod_name = os.path.splitext(f)[0]
                        tool_files[mod_name] = f

            # 检查 agent.py 的 _load_mcp_tools 函数中加载了哪些模块
            agent_path = os.path.join(self.package_root, "core", "agent.py")
            agent_content = self._read_file(agent_path) or ""
            loaded_modules = self._extract_loaded_mcp_modules(agent_content)

            # 检查 mcp_tools/__init__.py 中导出了哪些模块
            init_path = os.path.join(mcp_tools_dir, "__init__.py")
            init_content = self._read_file(init_path) or ""
            exported_modules = self._extract_init_imports(init_content)

            dead_files: List[Dict] = []
            alive_files: List[Dict] = []

            for mod_name, filename in tool_files.items():
                in_load = mod_name in loaded_modules
                in_init = mod_name in exported_modules

                if in_load or in_init:
                    ref_sources = []
                    if in_load:
                        ref_sources.append("_load_mcp_tools")
                    if in_init:
                        ref_sources.append("__init__")
                    alive_files.append({
                        "file": f"mcp_tools/{filename}",
                        "module": mod_name,
                        "referenced_in": " + ".join(ref_sources),
                    })
                else:
                    dead_files.append({
                        "file": f"mcp_tools/{filename}",
                        "module": mod_name,
                    })

            issues: List[Dict] = []
            for df in dead_files:
                issues.append({
                    "file": df["file"],
                    "line": 0,
                    "type": "dead_code",
                    "severity": "low",
                    "detail": (f"Module '{df['module']}' 未在 "
                               f"_load_mcp_tools 或 __init__.py 中被引用"),
                })

            data = {
                "dead_files": dead_files,
                "alive_files": alive_files,
                "total_scanned": len(tool_files),
            }

            self.log.info("detect_dead_code_done", data={
                "scanned": len(tool_files),
                "dead": len(dead_files),
                "alive": len(alive_files),
            })
            self.metrics.gauge("arch.dead_code.total_scanned",
                               len(tool_files))
            self.metrics.gauge("arch.dead_code.dead", len(dead_files))

            duration = time.perf_counter() - start
            return self._make_result(
                "detect_dead_code", success=True, data=data,
                issues=issues, duration=duration,
            )

    # ------------------------------------------------------------------
    # 6. review_version_consistency -- 版本号一致性检查
    # ------------------------------------------------------------------

    def review_version_consistency(self) -> AgentResult:
        """检查版本号一致性。

        - 读取 ``__init__.py`` 的 ``__version__``
        - 读取 ``setup.py`` 的 ``version=``
        - 读取 ``cli.py`` 的 ``version_option``
        - 读取 ``cli_tui.py`` 的 ``VERSION`` 常量
        - 比较一致性，推荐最高版本作为统一目标

        Returns:
            AgentResult, data 含:
                - versions: {"__init__": "x", "setup.py": "x", ...}
                - consistent: bool
                - recommended: 推荐统一版本号
        """
        start = time.perf_counter()
        with self._time_operation("review_version_consistency"):
            versions: Dict[str, str] = {}
            issues: List[Dict] = []

            # 1. __init__.py __version__
            init_path = os.path.join(self.package_root, "__init__.py")
            v = self._extract_version_assign(init_path, "__version__")
            if v:
                versions["__init__"] = v

            # 2. setup.py version=
            setup_path = os.path.join(self.project_root, "setup.py")
            v = self._extract_setup_version(setup_path)
            if v:
                versions["setup.py"] = v

            # 3. cli.py version_option
            cli_path = os.path.join(self.package_root, "cli.py")
            v = self._extract_cli_version_option(cli_path)
            if v:
                versions["cli.py"] = v

            # 4. cli_tui.py VERSION 常量
            cli_tui_path = os.path.join(self.package_root, "cli_tui.py")
            v = self._extract_version_assign(cli_tui_path, "VERSION")
            if v:
                versions["cli_tui.py"] = v

            # 比较一致性
            unique_versions = set(versions.values())
            consistent = len(unique_versions) <= 1

            # 推荐: 最高版本 (让所有源收敛到最高)
            if unique_versions:
                recommended = max(unique_versions,
                                  key=self._version_key)
            else:
                recommended = "0.1.0"

            if not consistent:
                for source, ver in versions.items():
                    if ver != recommended:
                        issues.append({
                            "file": source,
                            "line": 0,
                            "type": "version_mismatch",
                            "severity": "medium",
                            "detail": (f"{source}: {ver} "
                                       f"(推荐统一为 {recommended})"),
                        })

            data = {
                "versions": versions,
                "consistent": consistent,
                "recommended": recommended,
            }

            self.log.info("review_version_consistency_done", data={
                "versions": versions,
                "consistent": consistent,
                "recommended": recommended,
            })
            self.metrics.gauge("arch.version.consistent",
                               1 if consistent else 0)

            duration = time.perf_counter() - start
            return self._make_result(
                "review_version_consistency", success=True, data=data,
                issues=issues, duration=duration,
            )

    # ------------------------------------------------------------------
    # 7. enforce_architecture_redlines -- 架构红线综合检查
    # ------------------------------------------------------------------

    def enforce_architecture_redlines(self) -> AgentResult:
        """执行架构红线检查 -- 综合以上所有检查。

        - 调用 check_dag + audit_imports + audit_god_class
          + detect_dead_code + review_version_consistency
        - 汇总所有违规，按严重度排序
        - 返回汇总数据和合并的 issues 列表

        Returns:
            AgentResult, data 含:
                - total_violations: N
                - by_severity: {"critical": N, "high": N,
                                "medium": N, "low": N}
                - summary: 文字摘要
                - sub_checks: {方法名: 是否成功}
            issues 列表 (合并所有子检查的 issues)
        """
        start = time.perf_counter()
        with self._time_operation("enforce_architecture_redlines"):
            all_issues: List[Dict] = []
            sub_results: Dict[str, AgentResult] = {}

            checks = [
                ("check_dag", self.check_dag),
                ("audit_imports", self.audit_imports),
                ("audit_god_class", self.audit_god_class),
                ("detect_dead_code", self.detect_dead_code),
                ("review_version_consistency",
                 self.review_version_consistency),
            ]

            for name, check_fn in checks:
                try:
                    result = check_fn()
                    sub_results[name] = result
                    for issue in result.issues:
                        issue_copy = dict(issue)
                        issue_copy.setdefault("source", name)
                        issue_copy.setdefault("severity", "medium")
                        all_issues.append(issue_copy)
                except Exception as e:
                    self.log.error(f"{name}_failed",
                                   data={"error": str(e)})
                    all_issues.append({
                        "source": name,
                        "type": "check_error",
                        "severity": "high",
                        "detail": f"子检查 {name} 执行失败: {e}",
                    })

            # 按严重度统计
            by_severity = {
                "critical": 0, "high": 0, "medium": 0, "low": 0,
            }
            for issue in all_issues:
                sev = issue.get("severity", "medium")
                if sev in by_severity:
                    by_severity[sev] += 1
                else:
                    by_severity["medium"] += 1

            # 按严重度排序 (critical -> high -> medium -> low)
            all_issues.sort(
                key=lambda x: self._SEVERITY_ORDER.get(
                    x.get("severity", "medium"), 2),
            )

            total = len(all_issues)

            # 构建摘要
            summary_parts: List[str] = []
            for name, result in sub_results.items():
                issue_count = len(result.issues)
                summary_parts.append(f"{name}={issue_count}")
            for name in checks:
                pass
            # 补充失败的检查
            for name, _ in checks:
                if name not in sub_results:
                    summary_parts.append(f"{name}=ERROR")
            summary = "; ".join(summary_parts)

            data = {
                "total_violations": total,
                "by_severity": by_severity,
                "summary": summary,
                "sub_checks": {
                    name: r.success for name, r in sub_results.items()
                },
            }

            self.log.info("enforce_architecture_redlines_done", data={
                "total": total,
                "by_severity": by_severity,
            })
            self.metrics.gauge("arch.redlines.total_violations", total)
            for sev, count in by_severity.items():
                self.metrics.gauge(f"arch.redlines.{sev}", count)

            duration = time.perf_counter() - start
            return self._make_result(
                "enforce_architecture_redlines",
                success=True,
                data=data,
                issues=all_issues,
                duration=duration,
            )

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    # ---- 文件扫描 ----

    def _iter_py_files(self) -> List[str]:
        """遍历 package_root 下所有 .py 文件 (排除 __pycache__)。"""
        result: List[str] = []
        for root, dirs, files in os.walk(self.package_root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    result.append(os.path.join(root, f))
        return sorted(result)

    @staticmethod
    def _read_file(filepath: str) -> Optional[str]:
        """读取文件内容为字符串，失败返回 None。"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None

    def _read_ast(self, filepath: str) -> Optional[ast.AST]:
        """读取并解析 Python 文件为 AST，失败返回 None。"""
        content = self._read_file(filepath)
        if content is None:
            return None
        try:
            return ast.parse(content, filename=filepath)
        except SyntaxError as e:
            self.log.warning("ast_parse_failed",
                             data={"file": filepath, "error": str(e)})
            return None

    # ---- 模块分组 ----

    @staticmethod
    def _module_group(rel_path: str) -> str:
        """根据相对路径确定顶层模块组名。

        "core/agent.py" -> "core"
        "rag/retriever.py" -> "rag"
        "cli.py" -> "cli"
        "__init__.py" -> "__init__"
        """
        rel_path = rel_path.replace("\\", "/")
        parts = rel_path.split("/")
        if len(parts) > 1:
            return parts[0]
        filename = parts[0]
        if filename == "__init__.py":
            return "__init__"
        name, _ = os.path.splitext(filename)
        return name

    # ---- import 解析 ----

    def _resolve_import_from(
        self, node: ast.ImportFrom, source_group: str,
    ) -> Optional[str]:
        """将 ImportFrom 节点解析为目标模块组名。

        Returns:
            目标组名 (如 "core")，或 None 表示外部/同包导入。
        """
        module = node.module  # 可能为 None
        level = node.level    # 0=绝对, >0=相对

        if level > 0:
            # 相对 import
            if module is None:
                return None  # from . import X -> 同包
            parts = module.split(".")
            first = parts[0]
            if first in self._INTERNAL_GROUPS:
                return first
            return None  # 同包子模块
        else:
            # 绝对 import
            if module is None:
                return None
            if module.startswith("aerospace_agent."):
                parts = module.split(".")
                if len(parts) >= 2 and parts[1] in self._INTERNAL_GROUPS:
                    return parts[1]
                return None
            # 裸内部包名 (错误风格)
            parts = module.split(".")
            first = parts[0]
            if first in self._INTERNAL_GROUPS:
                return first
            return None

    def _resolve_import(self, name: str) -> Optional[str]:
        """将 Import 节点的名称解析为目标模块组名。

        "aerospace_agent.core.agent" -> "core"
        "os" -> None
        """
        if name.startswith("aerospace_agent."):
            parts = name.split(".")
            if len(parts) >= 2 and parts[1] in self._INTERNAL_GROUPS:
                return parts[1]
            return None
        parts = name.split(".")
        first = parts[0]
        if first in self._INTERNAL_GROUPS:
            return first
        return None

    def _is_bare_internal_import(self, module: str) -> bool:
        """检查模块路径是否为裸内部 import (如 'core.xxx')。

        排除以 'aerospace_agent' 开头的正确绝对 import。
        """
        if not module:
            return False
        if module.startswith("aerospace_agent"):
            return False
        parts = module.split(".")
        return parts[0] in self._INTERNAL_GROUPS

    # ---- sys.path hack 检测 ----

    @staticmethod
    def _is_sys_path_call(node: ast.Call) -> bool:
        """检查 Call 节点是否为 sys.path.insert() 或 sys.path.append()。"""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False
        if func.attr not in ("insert", "append"):
            return False
        # func.value 应为 sys.path
        inner = func.value
        if not isinstance(inner, ast.Attribute):
            return False
        if inner.attr != "path":
            return False
        if not (isinstance(inner.value, ast.Name)
                and inner.value.id == "sys"):
            return False
        return True

    @staticmethod
    def _get_call_attr(node: ast.Call) -> str:
        """获取 Attribute 调用的属性名 (如 'insert')。"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    # ---- 环检测 (三色标记 DFS) ----

    def _detect_cycles_dfs(
        self, adj: Dict[str, Set[str]],
    ) -> List[Dict]:
        """使用三色标记 DFS 检测有向图中的环。

        颜色定义:
            WHITE (0) -- 未访问
            GRAY  (1) -- 在当前 DFS 路径中 (正在探索)
            BLACK (2) -- 已完全探索

        当遇到 GRAY 节点时，说明存在回边 -> 环。
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {
            node: WHITE for node in adj
        }
        cycles: List[Dict] = []
        seen_keys: Set[str] = set()

        def dfs(node: str, stack: List[str]) -> None:
            color[node] = GRAY
            stack.append(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # 回边 -> 环
                    idx = stack.index(neighbor)
                    cycle_path = stack[idx:] + [neighbor]
                    key = self._normalize_cycle(cycle_path)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        cycles.append({
                            "path": cycle_path,
                            "type": "circular",
                        })
                elif color[neighbor] == WHITE:
                    dfs(neighbor, stack)
            stack.pop()
            color[node] = BLACK

        for node in list(adj.keys()):
            if color.get(node, BLACK) == WHITE:
                dfs(node, [])

        return cycles

    @staticmethod
    def _normalize_cycle(cycle: List[str]) -> str:
        """将环路径规范化以去重 (旋转到最小元素开头)。"""
        if len(cycle) <= 1:
            return "->".join(cycle)
        core = cycle[:-1]  # 去掉末尾重复节点
        if not core:
            return ""
        min_idx = core.index(min(core))
        rotated = core[min_idx:] + core[:min_idx]
        return "->".join(rotated)

    # ---- 上帝类检测辅助 ----

    @staticmethod
    def _count_class_attributes(node: ast.ClassDef) -> int:
        """统计类中 __init__ 方法里 self.xxx 属性的数量。"""
        attrs: Set[str] = set()
        for child in node.body:
            if (isinstance(child, ast.FunctionDef)
                    and child.name == "__init__"):
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if (isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"):
                                attrs.add(target.attr)
        return len(attrs)

    # ---- 配置统一辅助 ----

    def _extract_env_var(
        self, node: ast.Call, rel: str,
    ) -> Optional[Dict]:
        """从 os.environ.get() 或 os.getenv() 调用中提取环境变量信息。"""
        func = node.func

        # os.environ.get("VAR", default)
        if (isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"):
            return self._parse_env_call_args(node, rel)

        # os.getenv("VAR", default)
        if (isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"):
            return self._parse_env_call_args(node, rel)

        return None

    def _parse_env_call_args(
        self, node: ast.Call, rel: str,
    ) -> Optional[Dict]:
        """解析环境变量调用的参数。"""
        if not node.args:
            return None
        name_arg = node.args[0]
        if isinstance(name_arg, ast.Constant) and isinstance(
                name_arg.value, str):
            name = name_arg.value
        else:
            name = "<dynamic>"
        default: Any = None
        if len(node.args) >= 2:
            default = self._ast_to_value(node.args[1])
        return {
            "name": name,
            "default": default,
            "file": rel,
            "line": node.lineno,
        }

    @staticmethod
    def _ast_to_value(node: ast.AST) -> Any:
        """将 AST 常量节点转换为 Python 值。"""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        try:
            return ast.unparse(node)
        except Exception:
            return "<complex>"

    def _classify_config_group(self, var_name: str) -> str:
        """根据变量名关键词将环境变量归类到功能分组。"""
        upper = var_name.upper()
        for group, keywords in self._CONFIG_GROUPS.items():
            if any(kw in upper for kw in keywords):
                return group
        return "Other"

    def _generate_config_yaml(
        self,
        unique_vars: List[Dict],
        groups: Dict[str, List[Dict]],
    ) -> str:
        """生成 config.yaml 模板字符串。"""
        lines: List[str] = [
            "# ============================================",
            "# Aerospace Agent 统一配置",
            "# 由 ArchAgent.unify_config() 自动生成",
            f"# 共 {len(unique_vars)} 个环境变量",
            "# ============================================",
            "",
        ]
        all_groups = list(self._CONFIG_GROUPS.keys()) + ["Other"]
        for group in all_groups:
            members = groups.get(group, [])
            if not members:
                continue
            lines.append(f"# ---- {group} ----")
            lines.append(f"{group.lower()}:")
            for v in members:
                default = v.get("default")
                if default is None:
                    default_str = '""'
                elif isinstance(default, str):
                    default_str = f'"{default}"'
                elif isinstance(default, bool):
                    default_str = str(default).lower()
                else:
                    default_str = str(default)
                key = v["name"].lower()
                lines.append(
                    f"  {key}: {default_str}"
                    f"  # {v['file']}:{v['line']}"
                )
            lines.append("")
        return "\n".join(lines)

    # ---- 死代码检测辅助 ----

    @staticmethod
    def _extract_loaded_mcp_modules(content: str) -> Set[str]:
        """从 _load_mcp_tools 函数体中提取加载的模块名。

        查找函数体中的字符串常量 (如 "astro_dynamics_tool")，
        过滤出符合模块命名规则的字符串。
        """
        loaded: Set[str] = set()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return loaded

        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_load_mcp_tools"):
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Constant)
                            and isinstance(sub.value, str)):
                        val = sub.value
                        # 只接受符合模块命名规则的纯标识符
                        if re.match(r"^[a-z_][a-z0-9_]*$", val):
                            loaded.add(val)
        return loaded

    @staticmethod
    def _extract_init_imports(content: str) -> Set[str]:
        """从 __init__.py 中提取 import 的模块名。

        from .orekit_tool import OrekitTool -> "orekit_tool"
        from .base import BaseTool -> "base"
        """
        exported: Set[str] = set()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return exported

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = node.module.split(".")
                    exported.add(parts[-1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    exported.add(parts[-1])
        return exported

    # ---- 版本号检查辅助 ----

    def _extract_version_assign(
        self, filepath: str, var_name: str,
    ) -> Optional[str]:
        """从文件中提取变量赋值 (如 __version__ = '0.1.0')。

        优先用 AST 解析，失败时用正则回退。
        """
        tree = self._read_ast(filepath)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (isinstance(target, ast.Name)
                                and target.id == var_name
                                and isinstance(node.value, ast.Constant)
                                and isinstance(node.value.value, str)):
                            return node.value.value
        # 正则回退
        content = self._read_file(filepath)
        if content:
            pattern = rf'{var_name}\s*=\s*["\']([^"\']+)["\']'
            m = re.search(pattern, content)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_setup_version(filepath: str) -> Optional[str]:
        """从 setup.py 中提取 version= 参数。"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return None
        m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        return m.group(1) if m else None

    @staticmethod
    def _extract_cli_version_option(filepath: str) -> Optional[str]:
        """从 cli.py 中提取 @click.version_option(version='...')。"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return None
        m = re.search(
            r'version_option\s*\([^)]*version\s*=\s*["\']([^"\']+)["\']',
            content,
        )
        return m.group(1) if m else None

    @staticmethod
    def _version_key(v: str) -> tuple:
        """将版本字符串转换为可比较的元组。"""
        parts: list = []
        for p in v.split("."):
            try:
                parts.append(int(p))
            except ValueError:
                parts.append(0)
        return tuple(parts)


__all__ = ["ArchAgent"]
