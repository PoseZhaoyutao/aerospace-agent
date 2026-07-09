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
    ]


def build_basic_tools(workspace: Optional[Path] = None) -> List[BasicTool]:
    workspace = (workspace or Path.cwd()).resolve()

    def _write_text_file(path: str, content: str) -> Dict[str, Any]:
        return write_text_file(path=path, content=content, workspace=workspace)

    def _list_basic_tools() -> Dict[str, Any]:
        return {"status": "ok", "tools": _tool_catalog()}

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
    ]


def build_langchain_tools(workspace: Optional[Path] = None) -> List[Any]:
    """Build LangChain StructuredTool objects when langchain-core exists."""
    try:
        from langchain_core.tools import StructuredTool
    except ModuleNotFoundError:
        return []

    lc_tools = []
    for tool in build_basic_tools(workspace):

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
    ) -> None:
        self.llm = llm
        self.workspace = (workspace or Path.cwd()).resolve()
        self.config = config or BasicAgentConfig()
        self.tools = build_basic_tools(self.workspace)
        self.langchain_tools = build_langchain_tools(self.workspace)
        self.backend = "langchain-core" if self.langchain_tools else "builtin"
        self._runnable = _langchain_runnable(self._call_llm_once)

    def invoke(self, task: str) -> BasicAgentResult:
        task = task.strip()
        if self._is_static_site_request(task):
            return self._write_static_site(task)
        return self._llm_once(task)

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
        try:
            if self._runnable is not None:
                output = self._runnable.invoke(task)
            else:
                output = self._call_llm_once(task)
        except Exception as exc:
            return BasicAgentResult(
                ok=False,
                output=f"LLM 调用失败: {exc}",
                action="llm_once",
                backend=self.backend,
                errors=[str(exc)],
            )

        return BasicAgentResult(
            ok=True,
            output=str(output),
            action="llm_once",
            backend=self.backend,
        )

    def _call_llm_once(self, task: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是严格、务实的航天软件研究助理。"
                    "只做一次回答，不使用递归工具循环。"
                    "没有证据的内容必须标注为假设。"
                ),
            },
            {"role": "user", "content": task},
        ]
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
) -> BasicLangChainAgent:
    return BasicLangChainAgent(llm=llm, workspace=workspace, config=config)
