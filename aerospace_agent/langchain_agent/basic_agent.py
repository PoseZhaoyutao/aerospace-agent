"""Minimal LangChain-oriented agent facade.

This module is intentionally small. It provides a stable agent entrypoint for
basic workflows while keeping heavy ReAct, MCP orchestration, and skill routing
out of the default path.

If langchain-core is installed, the agent exposes LangChain tool/runnable
wrappers. If it is not installed, the same public API still works through the
built-in fallback. This is a factual fallback, not a claim that LangChain is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import html
import json
import re
import shutil
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class BasicTool:
    name: str
    description: str
    func: Callable[..., Dict[str, Any]]

    def invoke(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        return self.func(**payload)


@dataclass
class BasicAgentConfig:
    output_dir: str = "demo_output"
    max_output_tokens: int = 1024
    temperature: float = 0.2
    memory_window_messages: int = 12
    memory_max_chars: int = 12000
    rag_top_k: int = 3
    enable_rag_context: bool = True
    skill_install_dir: str = ".aerospace_skills"
    enable_skill_context: bool = True
    max_skill_contexts: int = 2
    skill_context_max_chars: int = 8000


@dataclass
class BasicAgentResult:
    ok: bool
    output: str
    action: str
    backend: str
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        return self.output


@dataclass
class SlidingWindowMemory:
    """In-memory conversation window for the basic LangChain path."""

    max_messages: int = 12
    max_chars: int = 12000
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported memory role: {role}")
        self.messages.append({"role": role, "content": str(content)})
        self._trim()

    def clear(self) -> None:
        self.messages.clear()

    def to_messages(self) -> List[Dict[str, str]]:
        self._trim()
        return [dict(message) for message in self.messages]

    def _trim(self) -> None:
        if self.max_messages > 0 and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        if self.max_chars <= 0:
            return

        total = 0
        kept: List[Dict[str, str]] = []
        for message in reversed(self.messages):
            size = len(message.get("content", ""))
            if kept and total + size > self.max_chars:
                break
            kept.append(message)
            total += size
        self.messages = list(reversed(kept))


def _workspace_path(path: str, workspace: Path) -> Path:
    raw = Path(path)
    target = raw if raw.is_absolute() else workspace / raw
    target = target.resolve()
    workspace = workspace.resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("PATH_OUTSIDE_WORKSPACE") from exc
    return target


def write_text_file(
    path: str,
    content: str,
    workspace: Optional[Path] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Write text inside the workspace and reject path traversal."""
    workspace = (workspace or Path.cwd()).resolve()
    try:
        target = _workspace_path(path, workspace)
    except ValueError:
        return {
            "status": "error",
            "error_code": "PATH_OUTSIDE_WORKSPACE",
            "path": path,
            "workspace": str(workspace),
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding=encoding)
    return {
        "status": "ok",
        "path": str(target),
        "bytes": len(content.encode(encoding, errors="replace")),
        "encoding": encoding,
    }


def _tool_catalog() -> List[Dict[str, str]]:
    return [
        {
            "name": "write_text_file",
            "description": "Write a UTF-8 text file under the current workspace.",
        },
        {
            "name": "list_basic_tools",
            "description": "Return the minimal built-in tool catalog.",
        },
        {
            "name": "list_mcp_tools",
            "description": "List tools exposed by the base MCP tool registry.",
        },
        {
            "name": "call_mcp_tool",
            "description": "Call a method on a base MCP tool by name.",
        },
        {
            "name": "list_skills",
            "description": "List registered Python skills and discovered SKILL.md manifests.",
        },
        {
            "name": "use_skill",
            "description": "Execute a registered Python skill by name.",
        },
        {
            "name": "discover_skill_manifests",
            "description": "Discover declarative SKILL.md manifests from local roots.",
        },
        {
            "name": "install_skill_from_path",
            "description": "Install a local directory containing SKILL.md into the workspace skill root.",
        },
        {
            "name": "run_terminal_command",
            "description": "Run a non-shell local command in the workspace as a fallback executor.",
        },
        {
            "name": "run_literature_keyword_cloud_workflow",
            "description": "Search/provide literature items and render keyword cloud artifacts.",
        },
        {
            "name": "index_orbit_dynamics_rag",
            "description": "Index the built-in orbit-dynamics expert seed corpus into RAG.",
        },
    ]


def _resolve_mcp_tools(mcp_tools: Optional[Any] = None) -> Dict[str, Any]:
    if mcp_tools is None:
        try:
            from aerospace_agent.mcp_tools.registry import list_tools
        except Exception:
            return {}
        try:
            return dict(list_tools())
        except Exception:
            return {}

    if isinstance(mcp_tools, dict):
        return dict(mcp_tools)

    resolved: Dict[str, Any] = {}
    for tool in mcp_tools:
        name = getattr(tool, "name", None)
        if name:
            resolved[name] = tool
    return resolved


def _resolve_skill_registry(
    skill_registry: Optional[Any] = None,
    skill_roots: Optional[List[Path]] = None,
) -> Any:
    if skill_registry is not None:
        if skill_roots:
            from aerospace_agent.skills.defaults import install_default_skill_manifests
            install_default_skill_manifests(skill_registry, roots=skill_roots)
        return skill_registry
    from aerospace_agent.skills.registry import SkillRegistry
    from aerospace_agent.skills.defaults import default_skill_roots, install_default_skill_manifests
    roots = skill_roots if skill_roots is not None else default_skill_roots()
    registry = SkillRegistry(skill_roots=roots)
    try:
        registry.auto_discover()
    except Exception:
        pass
    try:
        install_default_skill_manifests(registry, roots=roots)
    except Exception:
        pass
    return registry


def _safe_skill_dir_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name).strip()).strip(".-")
    return safe or "skill"


def _as_path_list(values: Optional[Any]) -> List[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        return [Path(values)]
    return [Path(value) for value in values]


def build_basic_tools(
    workspace: Optional[Path] = None,
    mcp_tools: Optional[Any] = None,
    skill_registry: Optional[Any] = None,
    skill_agent: Optional[Any] = None,
    skill_roots: Optional[Any] = None,
    skill_install_dir: Optional[Path] = None,
    rag: Optional[Any] = None,
) -> List[BasicTool]:
    workspace = (workspace or Path.cwd()).resolve()
    skill_root_paths = _as_path_list(skill_roots)
    install_root = (
        Path(skill_install_dir)
        if skill_install_dir is not None
        else workspace / ".aerospace_skills"
    ).resolve()
    registry = _resolve_skill_registry(skill_registry, skill_root_paths)

    def _write_text_file(path: str, content: str) -> Dict[str, Any]:
        return write_text_file(path=path, content=content, workspace=workspace)

    def _list_basic_tools() -> Dict[str, Any]:
        return {"status": "ok", "tools": _tool_catalog()}

    def _list_mcp_tools() -> Dict[str, Any]:
        tools = _resolve_mcp_tools(mcp_tools)
        return {
            "status": "ok",
            "tools": [
                (
                    tool.get_info()
                    if hasattr(tool, "get_info")
                    else {
                        "name": name,
                        "description": getattr(tool, "description", ""),
                        "source": getattr(tool, "source", "unknown"),
                        "methods": (
                            tool.list_methods()
                            if hasattr(tool, "list_methods")
                            else list(getattr(tool, "methods_schema", {}).keys())
                        ),
                    }
                )
                for name, tool in tools.items()
            ],
        }

    def _call_mcp_tool(
        name: str,
        method: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tools = _resolve_mcp_tools(mcp_tools)
        tool = tools.get(name)
        if tool is None:
            return {
                "status": "error",
                "error_code": "MCP_TOOL_NOT_FOUND",
                "tool": name,
                "available": list(tools.keys()),
            }
        if not method:
            return {
                "status": "error",
                "error_code": "MISSING_METHOD",
                "tool": name,
                "methods": (
                    tool.list_methods()
                    if hasattr(tool, "list_methods")
                    else list(getattr(tool, "methods_schema", {}).keys())
                ),
            }
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return {
                "status": "error",
                "error_code": "INVALID_ARGUMENTS",
                "tool": name,
                "method": method,
            }
        try:
            result = tool.call(method, **arguments)
        except Exception as exc:
            return {
                "status": "error",
                "error_code": "MCP_CALL_FAILED",
                "tool": name,
                "method": method,
                "error": str(exc),
            }
        return {
            "status": "ok",
            "tool": name,
            "method": method,
            "result": result,
        }

    def _list_skills(category: Optional[str] = None) -> Dict[str, Any]:
        skills = registry.list_skills(category=category)
        manifests = registry.list_skill_manifests(category=category)
        return {
            "status": "ok",
            "skills": skills,
            "manifests": manifests,
            "count": len(skills) + len(manifests),
        }

    def _use_skill(
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return {
                "status": "error",
                "error_code": "INVALID_ARGUMENTS",
                "skill": name,
            }
        result = registry.execute(skill_agent, name, **arguments)
        if result.get("error_code") == "SKILL_NOT_EXECUTABLE":
            manifest = registry.get_manifest(name)
            if manifest:
                try:
                    instructions = Path(manifest["path"]).read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    instructions = Path(manifest["path"]).read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                except Exception as exc:
                    return {
                        "status": "error",
                        "skill": name,
                        "result": {
                            "success": False,
                            "error_code": "SKILL_INSTRUCTIONS_UNREADABLE",
                            "message": str(exc),
                        },
                    }
                return {
                    "status": "ok",
                    "skill": name,
                    "result": {
                        "success": True,
                        "execution_mode": "instruction_context",
                        "manifest": manifest,
                        "instructions": instructions,
                        "message": (
                            "Declarative skill loaded as instructions. "
                            "Use run_terminal_command/write_text_file for fallback execution."
                        ),
                    },
                }
        return {
            "status": "ok" if result.get("success") else "error",
            "skill": name,
            "result": result,
        }

    def _discover_skill_manifests(
        roots: Optional[Any] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        scan_roots = _as_path_list(roots) or skill_root_paths or [install_root]
        try:
            count = registry.discover_manifests(scan_roots)
        except Exception as exc:
            return {
                "status": "error",
                "error_code": "SKILL_DISCOVERY_FAILED",
                "error": str(exc),
            }
        return {
            "status": "ok",
            "count": count,
            "roots": [str(root) for root in scan_roots],
            "manifests": registry.list_skill_manifests(category=category),
        }

    def _install_skill_from_path(
        path: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        from aerospace_agent.skills.manifest import validate_skill_manifest

        source = Path(path).resolve()
        if source.is_file() and source.name == "SKILL.md":
            source_dir = source.parent
            manifest_path = source
        else:
            source_dir = source
            manifest_path = source_dir / "SKILL.md"

        if not manifest_path.is_file():
            return {
                "status": "error",
                "error_code": "MISSING_SKILL_MD",
                "path": str(source),
            }

        manifest = validate_skill_manifest(manifest_path, root=source_dir)
        target_name = _safe_skill_dir_name(manifest.get("name") or source_dir.name)
        target_dir = install_root / target_name

        if target_dir.exists() and not overwrite:
            return {
                "status": "error",
                "error_code": "SKILL_ALREADY_INSTALLED",
                "installed_path": str(target_dir),
                "manifest": manifest,
            }

        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=overwrite)
        except Exception as exc:
            return {
                "status": "error",
                "error_code": "SKILL_INSTALL_FAILED",
                "error": str(exc),
                "path": str(source),
            }

        installed_manifest = validate_skill_manifest(
            target_dir / "SKILL.md",
            root=install_root,
        )
        if install_root not in getattr(registry, "skill_roots", []):
            try:
                registry.skill_roots.append(install_root)
            except Exception:
                pass
        try:
            registry.discover_manifests([install_root])
        except Exception:
            pass
        return {
            "status": "ok",
            "installed_path": str(target_dir),
            "manifest": installed_manifest,
        }

    def _run_terminal_command(
        cmd: List[str],
        timeout: float = 60,
        max_output_chars: int = 12000,
    ) -> Dict[str, Any]:
        if isinstance(cmd, str) or not isinstance(cmd, list) or not cmd:
            return {
                "status": "error",
                "error_code": "INVALID_COMMAND",
                "message": "cmd must be a non-empty list; shell strings are not accepted.",
            }
        from aerospace_agent.local_runtime import run_command
        result = run_command(
            cmd,
            cwd=workspace,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        return {
            "status": "ok" if result.ok else "error",
            "cmd": result.cmd,
            "cwd": result.cwd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timeout": result.timeout,
            "encoding": result.encoding,
        }

    def _run_literature_keyword_cloud_workflow(
        query: str,
        papers: Optional[List[Any]] = None,
        output_dir: str = "artifacts/literature_keyword_cloud",
        max_results: int = 10,
        max_keywords: int = 30,
    ) -> Dict[str, Any]:
        from aerospace_agent.workflows.literature_keyword_cloud import (
            run_literature_keyword_cloud_workflow,
        )
        try:
            out = _workspace_path(str(output_dir), workspace)
        except ValueError:
            return {
                "status": "error",
                "error_code": "PATH_OUTSIDE_WORKSPACE",
                "output_dir": output_dir,
                "workspace": str(workspace),
            }
        return run_literature_keyword_cloud_workflow(
            query=query,
            papers=papers,
            rag=rag,
            output_dir=out,
            max_results=max_results,
            max_keywords=max_keywords,
        )

    def _index_orbit_dynamics_rag() -> Dict[str, Any]:
        from aerospace_agent.rag.orbit_dynamics import index_orbit_dynamics_corpus
        return index_orbit_dynamics_corpus(rag)

    return [
        BasicTool(
            name="write_text_file",
            description="Write a UTF-8 text file under the current workspace.",
            func=_write_text_file,
        ),
        BasicTool(
            name="list_basic_tools",
            description="Return the minimal built-in tool catalog.",
            func=_list_basic_tools,
        ),
        BasicTool(
            name="list_mcp_tools",
            description="List tools exposed by the base MCP tool registry.",
            func=_list_mcp_tools,
        ),
        BasicTool(
            name="call_mcp_tool",
            description="Call a method on a base MCP tool by name.",
            func=_call_mcp_tool,
        ),
        BasicTool(
            name="list_skills",
            description="List registered Python skills and discovered SKILL.md manifests.",
            func=_list_skills,
        ),
        BasicTool(
            name="use_skill",
            description="Execute a registered Python skill by name.",
            func=_use_skill,
        ),
        BasicTool(
            name="discover_skill_manifests",
            description="Discover declarative SKILL.md manifests from local roots.",
            func=_discover_skill_manifests,
        ),
        BasicTool(
            name="install_skill_from_path",
            description="Install a local directory containing SKILL.md into the workspace skill root.",
            func=_install_skill_from_path,
        ),
        BasicTool(
            name="run_terminal_command",
            description="Run a non-shell local command in the workspace as a fallback executor.",
            func=_run_terminal_command,
        ),
        BasicTool(
            name="run_literature_keyword_cloud_workflow",
            description="Search/provide literature items and render keyword cloud artifacts.",
            func=_run_literature_keyword_cloud_workflow,
        ),
        BasicTool(
            name="index_orbit_dynamics_rag",
            description="Index the built-in orbit-dynamics expert seed corpus into RAG.",
            func=_index_orbit_dynamics_rag,
        ),
    ]


def build_langchain_tools(
    workspace: Optional[Path] = None,
    mcp_tools: Optional[Any] = None,
    skill_registry: Optional[Any] = None,
    skill_agent: Optional[Any] = None,
    skill_roots: Optional[Any] = None,
    skill_install_dir: Optional[Path] = None,
    rag: Optional[Any] = None,
) -> List[Any]:
    """Build LangChain StructuredTool objects when langchain-core exists."""
    try:
        from langchain_core.tools import StructuredTool
    except ModuleNotFoundError:
        return []

    lc_tools = []
    for tool in build_basic_tools(
        workspace,
        mcp_tools=mcp_tools,
        skill_registry=skill_registry,
        skill_agent=skill_agent,
        skill_roots=skill_roots,
        skill_install_dir=skill_install_dir,
        rag=rag,
    ):

        def _run(_tool: BasicTool = tool, **kwargs: Any) -> str:
            return json.dumps(_tool.invoke(kwargs), ensure_ascii=False, default=str)

        lc_tools.append(
            StructuredTool.from_function(
                func=_run,
                name=tool.name,
                description=tool.description,
            )
        )
    return lc_tools


def _langchain_runnable(callback: Callable[[str], str]) -> Optional[Any]:
    try:
        from langchain_core.runnables import RunnableLambda
    except ModuleNotFoundError:
        return None
    return RunnableLambda(callback)


class BasicLangChainAgent:
    """A minimal agent entrypoint with an optional LangChain Core backend.

    Guarantees:
    - no recursive ReAct loop in the default path
    - no automatic use of the full project tool registry
    - deterministic handling for simple local file/site generation requests
    - one LLM call for ordinary prompts
    """

    def __init__(
        self,
        llm: Any,
        workspace: Optional[Path] = None,
        config: Optional[BasicAgentConfig] = None,
        memory: Optional[SlidingWindowMemory] = None,
        rag: Optional[Any] = None,
        mcp_tools: Optional[Any] = None,
        skill_registry: Optional[Any] = None,
        skill_agent: Optional[Any] = None,
        skill_roots: Optional[Any] = None,
        skill_install_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.workspace = (workspace or Path.cwd()).resolve()
        self.config = config or BasicAgentConfig()
        self.memory = memory or SlidingWindowMemory(
            max_messages=self.config.memory_window_messages,
            max_chars=self.config.memory_max_chars,
        )
        self.rag = rag
        self.mcp_tools = mcp_tools
        self.skill_registry = _resolve_skill_registry(
            skill_registry,
            _as_path_list(skill_roots),
        )
        self.skill_agent = skill_agent
        self.skill_roots = skill_roots
        self.skill_install_dir = (
            Path(skill_install_dir)
            if skill_install_dir is not None
            else self.workspace / self.config.skill_install_dir
        ).resolve()
        self.tools = build_basic_tools(
            self.workspace,
            mcp_tools=self.mcp_tools,
            skill_registry=self.skill_registry,
            skill_agent=self.skill_agent,
            skill_roots=self.skill_roots,
            skill_install_dir=self.skill_install_dir,
            rag=self.rag,
        )
        self.langchain_tools = build_langchain_tools(
            self.workspace,
            mcp_tools=self.mcp_tools,
            skill_registry=self.skill_registry,
            skill_agent=self.skill_agent,
            skill_roots=self.skill_roots,
            skill_install_dir=self.skill_install_dir,
            rag=self.rag,
        )
        self.backend = "langchain-core" if self.langchain_tools else "builtin"
        self._runnable = _langchain_runnable(self._call_llm_once)

    def invoke(self, task: str) -> BasicAgentResult:
        task = task.strip()
        direct_skill_result = self._handle_direct_skill_request(task)
        if direct_skill_result is not None:
            self._remember(task, direct_skill_result.output)
            return direct_skill_result
        if self._is_static_site_request(task):
            result = self._write_static_site(task)
            self._remember(task, result.output)
            return result
        return self._llm_once(task)

    def set_interfaces(
        self,
        rag: Optional[Any] = None,
        mcp_tools: Optional[Any] = None,
        skill_registry: Optional[Any] = None,
        skill_agent: Optional[Any] = None,
    ) -> None:
        self.rag = rag
        self.mcp_tools = mcp_tools
        if skill_registry is not None:
            self.skill_registry = skill_registry
        if skill_agent is not None:
            self.skill_agent = skill_agent
        self.tools = build_basic_tools(
            self.workspace,
            mcp_tools=self.mcp_tools,
            skill_registry=self.skill_registry,
            skill_agent=self.skill_agent,
            skill_roots=self.skill_roots,
            skill_install_dir=self.skill_install_dir,
            rag=self.rag,
        )
        self.langchain_tools = build_langchain_tools(
            self.workspace,
            mcp_tools=self.mcp_tools,
            skill_registry=self.skill_registry,
            skill_agent=self.skill_agent,
            skill_roots=self.skill_roots,
            skill_install_dir=self.skill_install_dir,
            rag=self.rag,
        )
        self.backend = "langchain-core" if self.langchain_tools else "builtin"

    @staticmethod
    def _is_static_site_request(task: str) -> bool:
        wants_page = any(word in task for word in ("网站", "网页", "HTML", "html", "页面"))
        wants_file = any(word in task for word in ("保存", "本地", "写", "生成", "创建"))
        return wants_page and wants_file

    def _extract_output_path(self, task: str) -> str:
        patterns = [
            r"(?:保存到|保存为|写到)\s*([^\s，。]+\.html?)",
            r"([A-Za-z0-9_\-./\\]+\.html?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task)
            if match:
                return match.group(1).replace("\\", "/")
        return f"{self.config.output_dir}/site/index.html"

    def _write_static_site(self, task: str) -> BasicAgentResult:
        output_path = self._extract_output_path(task)
        title = "花里胡哨的网站" if "花里胡哨" in task else "本地静态网站"
        content = self._render_static_site(title=title, task=task)
        tool = self._tool("write_text_file")
        result = tool.invoke({"path": output_path, "content": content})
        if result.get("status") != "ok":
            return BasicAgentResult(
                ok=False,
                output=f"写入失败: {result.get('error_code', 'UNKNOWN_ERROR')}",
                action="write_static_site",
                backend=self.backend,
                errors=[json.dumps(result, ensure_ascii=False, default=str)],
            )

        path = result["path"]
        return BasicAgentResult(
            ok=True,
            output=(
                "已生成静态网页。\n"
                f"文件: {path}\n"
                "执行路径: basic_intent_router -> write_text_file\n"
                "说明: 该基础任务未进入递归 ReAct 循环。"
            ),
            action="write_static_site",
            backend=self.backend,
            artifacts=[path],
            metadata={"tool_result": result},
        )

    def _llm_once(self, task: str) -> BasicAgentResult:
        skill_contexts = self._skill_contexts_for_task(task)
        try:
            if self._runnable is not None:
                output = self._call_llm_once(task, skill_contexts=skill_contexts)
            else:
                output = self._call_llm_once(task, skill_contexts=skill_contexts)
        except Exception as exc:
            return BasicAgentResult(
                ok=False,
                output=f"LLM 调用失败: {exc}",
                action="llm_once",
                backend=self.backend,
                errors=[str(exc)],
            )

        result = BasicAgentResult(
            ok=True,
            output=str(output),
            action="llm_once",
            backend=self.backend,
            metadata={
                "skills": [
                    {
                        "name": context["name"],
                        "path": context.get("path"),
                        "execution_mode": context.get("execution_mode"),
                    }
                    for context in skill_contexts
                ],
            } if skill_contexts else {},
        )
        self._remember(task, result.output)
        return result

    def _call_llm_once(
        self,
        task: str,
        skill_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        messages = self._build_messages(task, skill_contexts=skill_contexts)
        if hasattr(self.llm, "chat"):
            return str(
                self.llm.chat(
                    messages,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                )
            )
        if callable(self.llm):
            return str(self.llm(messages))
        raise RuntimeError("LLM_UNAVAILABLE")

    def _build_messages(
        self,
        task: str,
        skill_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是严格、务实的航天软件研究助理。"
                    "只做一次回答，不使用递归工具循环。"
                    "没有证据的内容必须标注为假设。"
                ),
            },
        ]
        skill_context_text = self._format_skill_contexts(
            skill_contexts if skill_contexts is not None else self._skill_contexts_for_task(task)
        )
        if skill_context_text:
            messages.append({
                "role": "system",
                "content": skill_context_text,
            })
        rag_context = self._rag_context(task)
        if rag_context:
            messages.append({
                "role": "system",
                "content": "RAG context (unverified evidence, not a conclusion):\n" + rag_context,
            })
        messages.extend(self.memory.to_messages())
        messages.append({"role": "user", "content": task})
        return messages

    def _remember(self, task: str, output: str) -> None:
        self.memory.add("user", task)
        self.memory.add("assistant", output)

    def _handle_direct_skill_request(self, task: str) -> Optional[BasicAgentResult]:
        lowered = task.lower()
        compact = re.sub(r"\s+", "", lowered)

        if any(phrase in compact for phrase in ("列出技能", "查看技能", "有哪些技能")) or "list skills" in lowered:
            tool_result = self._tool("list_skills").invoke()
            return BasicAgentResult(
                ok=tool_result.get("status") == "ok",
                output=json.dumps(tool_result, ensure_ascii=False, indent=2, default=str),
                action="list_skills",
                backend=self.backend,
                metadata={"tool_result": tool_result},
            )

        install_path = self._extract_install_skill_path(task)
        if install_path:
            tool_result = self._tool("install_skill_from_path").invoke({
                "path": install_path,
                "overwrite": True,
            })
            return BasicAgentResult(
                ok=tool_result.get("status") == "ok",
                output=json.dumps(tool_result, ensure_ascii=False, indent=2, default=str),
                action="install_skill_from_path",
                backend=self.backend,
                metadata={"tool_result": tool_result},
            )

        skill_name = self._extract_direct_load_skill_name(task)
        if skill_name:
            tool_result = self._tool("use_skill").invoke({"name": skill_name})
            return BasicAgentResult(
                ok=tool_result.get("status") == "ok",
                output=self._format_direct_skill_output(skill_name, tool_result, task),
                action="use_skill",
                backend=self.backend,
                metadata={"tool_result": tool_result},
            )
        return None

    @staticmethod
    def _extract_install_skill_path(task: str) -> Optional[str]:
        lowered = task.lower()
        if "install skill" not in lowered and not ("安装" in task and "技能" in task):
            return None
        quoted = re.search(r'["“](.+?)["”]', task)
        if quoted:
            return quoted.group(1).strip()
        match = re.search(r"(?:install skill|安装技能|安装\s+skill)\s+(.+)$", task, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_direct_load_skill_name(self, task: str) -> Optional[str]:
        lowered = task.lower()
        if "use_skill" in lowered:
            match = re.search(r"use_skill\s*\(?\s*['\"]?([A-Za-z0-9_.:-]+)", task)
            if match:
                return self._resolve_skill_name(match.group(1))

        load_intent = (
            any(word in task for word in ("加载", "读取", "查看", "显示", "使用", "调用"))
            or ("用" in task and "技能" in task)
            or ("load" in lowered and "skill" in lowered)
            or ("show" in lowered and "skill" in lowered)
            or ("use" in lowered and "skill" in lowered)
        )
        if not load_intent or "技能" not in task and "skill" not in lowered:
            return None

        for name in self._all_skill_names():
            if self._task_mentions_skill(task, name):
                return name
        return None

    def _format_direct_skill_output(
        self,
        skill_name: str,
        tool_result: Dict[str, Any],
        task: str,
    ) -> str:
        status = tool_result.get("status", "unknown")
        payload = tool_result.get("result") or {}
        lines = [
            f"skill: {skill_name}",
            f"status: {status}",
            f"execution_mode: {payload.get('execution_mode', 'python_executor')}",
        ]
        manifest = payload.get("manifest") or {}
        if manifest.get("path"):
            lines.append(f"path: {manifest['path']}")

        if self._is_pdf_skill_without_pdf_path(skill_name, task):
            lines.append(
                "未提供 PDF 文件路径，不能执行版式检查；当前只完成了 pdf 技能加载。"
            )
            lines.append(
                "请提供本地 .pdf 路径后再执行，例如：使用 pdf 技能检查 D:\\path\\paper.pdf"
            )

        instructions = str(payload.get("instructions") or "")
        if instructions:
            lines.append("")
            lines.append("instructions:")
            lines.append(instructions)
        else:
            lines.append("")
            lines.append("raw_result:")
            lines.append(json.dumps(tool_result, ensure_ascii=False, indent=2, default=str))
        return "\n".join(lines)

    @staticmethod
    def _is_pdf_skill_without_pdf_path(skill_name: str, task: str) -> bool:
        if skill_name.lower().split(":")[-1] != "pdf":
            return False
        wants_check = any(word in task.lower() for word in ("检查", "审查", "查看", "inspect", "review", "check"))
        has_pdf_path = re.search(r"(?i)(?:[A-Za-z]:[\\/]|\.{0,2}[\\/]|/)?[^\s，。；;]+\.pdf\b", task) is not None
        return wants_check and not has_pdf_path

    def _skill_contexts_for_task(self, task: str) -> List[Dict[str, Any]]:
        if not self.config.enable_skill_context:
            return []

        contexts: List[Dict[str, Any]] = []
        for name in self._skill_names_for_task(task):
            tool_result = self._tool("use_skill").invoke({"name": name})
            payload = tool_result.get("result") or {}
            if tool_result.get("status") != "ok":
                continue
            if payload.get("execution_mode") != "instruction_context":
                continue
            instructions = str(payload.get("instructions") or "")
            if not instructions:
                continue
            manifest = payload.get("manifest") or {}
            contexts.append({
                "name": name,
                "path": manifest.get("path"),
                "execution_mode": payload.get("execution_mode"),
                "instructions": instructions[: self.config.skill_context_max_chars],
            })
            if len(contexts) >= max(1, self.config.max_skill_contexts):
                break
        return contexts

    def _skill_names_for_task(self, task: str) -> List[str]:
        names: List[str] = []
        for name in self._all_skill_names():
            if self._task_mentions_skill(task, name):
                names.append(name)
        return names

    def _all_skill_names(self) -> List[str]:
        items = []
        try:
            items.extend(self.skill_registry.list_skills())
        except Exception:
            pass
        try:
            items.extend(self.skill_registry.list_skill_manifests())
        except Exception:
            pass

        names = []
        seen = set()
        for item in items:
            name = str(item.get("name", "")).strip()
            key = name.lower()
            if name and key not in seen:
                names.append(name)
                seen.add(key)
        return names

    def _resolve_skill_name(self, requested: str) -> str:
        requested_key = requested.lower()
        for name in self._all_skill_names():
            if name.lower() == requested_key:
                return name
        return requested

    @staticmethod
    def _task_mentions_skill(task: str, name: str) -> bool:
        lowered = task.lower()
        skill = name.lower()
        variants = {
            skill,
            skill.replace("-", " "),
            skill.replace("_", " "),
            skill.split(":")[-1],
        }
        normalized_task = re.sub(r"[^a-z0-9]+", " ", lowered)
        for variant in variants:
            if not variant:
                continue
            if f"${variant}" in lowered or f"@{variant}" in lowered:
                return True
            normalized_variant = re.sub(r"[^a-z0-9]+", " ", variant).strip()
            if normalized_variant and re.search(rf"(?<![a-z0-9]){re.escape(normalized_variant)}(?![a-z0-9])", normalized_task):
                return True
        return False

    @staticmethod
    def _format_skill_contexts(contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return ""
        blocks = []
        for context in contexts:
            header = (
                "Skill context (local operating instructions, not evidence):\n"
                f"Skill: {context['name']}\n"
                f"Path: {context.get('path') or 'unknown'}\n"
                f"Execution mode: {context.get('execution_mode')}\n"
            )
            blocks.append(header + str(context.get("instructions") or ""))
        return "\n\n---\n\n".join(blocks)

    def _rag_context(self, task: str) -> str:
        if not self.config.enable_rag_context or self.rag is None:
            return ""

        for method_name in ("retrieve", "search", "recall"):
            method = getattr(self.rag, method_name, None)
            if method is None:
                continue
            try:
                try:
                    result = method(task, top_k=self.config.rag_top_k)
                except TypeError:
                    result = method(task)
            except Exception:
                continue
            return self._format_rag_result(result)
        return ""

    @staticmethod
    def _format_rag_result(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result[:4000]
        if isinstance(result, dict):
            for key in ("text", "content", "result", "answer"):
                if key in result:
                    return str(result[key])[:4000]
            return json.dumps(result, ensure_ascii=False, default=str)[:4000]
        if isinstance(result, list):
            lines = []
            for item in result[:5]:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("result")
                    source = item.get("source") or item.get("path") or item.get("id")
                    if text:
                        prefix = f"[{source}] " if source else ""
                        lines.append(prefix + str(text))
                    else:
                        lines.append(json.dumps(item, ensure_ascii=False, default=str))
                else:
                    lines.append(str(item))
            return "\n".join(lines)[:4000]
        return str(result)[:4000]

    def _tool(self, name: str) -> BasicTool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(name)

    @staticmethod
    def _render_static_site(title: str, task: str) -> str:
        safe_title = html.escape(title)
        safe_task = html.escape(task)
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111318;
      --fg: #f7f7fb;
      --cyan: #3ee7d6;
      --pink: #ff5ca8;
      --gold: #ffd166;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      overflow: hidden;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--fg);
      background:
        radial-gradient(circle at 20% 20%, rgba(62,231,214,.28), transparent 30%),
        radial-gradient(circle at 80% 30%, rgba(255,92,168,.25), transparent 34%),
        linear-gradient(135deg, #111318, #251a39 55%, #10282f);
    }}
    main {{
      width: min(900px, calc(100vw - 32px));
      padding: clamp(28px, 6vw, 68px);
      border: 1px solid rgba(255,255,255,.18);
      background: rgba(10,12,18,.62);
      backdrop-filter: blur(18px);
      box-shadow: 0 24px 80px rgba(0,0,0,.45);
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: clamp(40px, 8vw, 92px);
      line-height: .95;
      letter-spacing: 0;
    }}
    p {{
      max-width: 680px;
      margin: 0 0 28px;
      font-size: clamp(16px, 2vw, 22px);
      line-height: 1.7;
      color: rgba(247,247,251,.82);
    }}
    .actions {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    button {{
      border: 0;
      padding: 12px 18px;
      color: #111318;
      background: var(--gold);
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{ background: var(--cyan); }}
    .spark {{
      position: fixed;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--pink);
      animation: float 6s linear infinite;
      opacity: .8;
    }}
    @keyframes float {{
      from {{ transform: translateY(110vh) rotate(0deg); }}
      to {{ transform: translateY(-20vh) rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <p>这是根据请求生成的本地静态网页: {safe_task}</p>
    <div class="actions">
      <button onclick="document.body.style.filter='hue-rotate(90deg)'">换个颜色</button>
      <button class="secondary" onclick="alert('本页已保存到本地 HTML 文件。')">查看状态</button>
    </div>
  </main>
  <script>
    for (let i = 0; i < 28; i++) {{
      const node = document.createElement('i');
      node.className = 'spark';
      node.style.left = Math.random() * 100 + 'vw';
      node.style.animationDelay = Math.random() * 6 + 's';
      node.style.animationDuration = 4 + Math.random() * 5 + 's';
      document.body.appendChild(node);
    }}
  </script>
</body>
</html>
"""


def create_basic_langchain_agent(
    llm: Any,
    workspace: Optional[Path] = None,
    config: Optional[BasicAgentConfig] = None,
    memory: Optional[SlidingWindowMemory] = None,
    rag: Optional[Any] = None,
    mcp_tools: Optional[Any] = None,
    skill_registry: Optional[Any] = None,
    skill_agent: Optional[Any] = None,
    skill_roots: Optional[Any] = None,
    skill_install_dir: Optional[Path] = None,
) -> BasicLangChainAgent:
    return BasicLangChainAgent(
        llm=llm,
        workspace=workspace,
        config=config,
        memory=memory,
        rag=rag,
        mcp_tools=mcp_tools,
        skill_registry=skill_registry,
        skill_agent=skill_agent,
        skill_roots=skill_roots,
        skill_install_dir=skill_install_dir,
    )
