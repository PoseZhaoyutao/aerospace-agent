"""aerospace-agent CLI 交互终端（LangChain 基础 Agent）。

核心设计：
    1. LangChain 基础入口：短上下文、滑动窗口记忆、非递归工具路径
    2. 保留 RAG / MCP 接口，但不把旧 ReAct/fast/CEO 模式暴露给用户
    3. CodeWhale 风格布局：Header → Transcript → Footer → Composer

用法::
    python -m aerospace_agent.cli_tui              # 默认连接 Qwen3
    python -m aerospace_agent.cli_tui --mock       # 离线 MockLLM
"""
from __future__ import annotations

import os
import sys
import time
import json
import io
import re
import ast
import math
import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.table import Table
    from rich.markdown import Markdown
    from rich import box
except ImportError:
    print("需要安装 rich：pip install rich")
    sys.exit(1)

VERSION = "0.7.0"
QWEN3_URL = "http://127.0.0.1:8000/v1"
QWEN3_MODEL = "qwen3-vl"


# ======================================================================
# 文件夹权限系统
# ======================================================================

class PermissionLevel:
    """文件夹编辑权限级别。"""
    NONE = "none"       # 禁止编辑文件
    ASK = "ask"         # 每次编辑前询问
    AUTO = "auto"       # 全自动审批

    ALL = [NONE, ASK, AUTO]
    LABELS = {
        NONE: "禁止编辑",
        ASK: "询问后编辑",
        AUTO: "全自动审批",
    }
    COLORS = {
        NONE: "red",
        ASK: "yellow",
        AUTO: "green",
    }


@dataclass
class FolderPermission:
    """单个文件夹的权限状态。"""
    path: str = ""
    level: str = PermissionLevel.NONE

    def is_writable(self) -> bool:
        return self.level != PermissionLevel.NONE

    def should_ask(self) -> bool:
        return self.level == PermissionLevel.ASK


# ======================================================================
# 统计追踪
# ======================================================================

class Stats:
    """任务统计。"""

    def __init__(self):
        self.reset_task()
        self.history: List[Dict] = []
        self.total_tokens = 0
        self.total_time = 0.0

    def reset_task(self):
        self.start = 0.0
        self.end = 0.0
        self.llm_calls = 0
        self.tool_calls = 0
        self.react_steps = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def begin(self):
        self.reset_task()
        self.start = time.time()

    def finish(self, success: bool):
        self.end = time.time()
        elapsed = self.elapsed
        self.total_time += elapsed
        self.total_tokens += self.in_tokens + self.out_tokens
        self.history.append({
            "success": success, "time": round(elapsed, 2),
            "tokens": self.in_tokens + self.out_tokens,
            "llm": self.llm_calls, "tools": self.tool_calls,
            "steps": self.react_steps,
        })

    def llm_call(self, text_in: str, text_out: str):
        self.llm_calls += 1
        self.in_tokens += max(1, len(text_in) // 3)
        self.out_tokens += max(1, len(text_out) // 3)

    @property
    def elapsed(self) -> float:
        if not self.start:
            return 0.0
        return (self.end or time.time()) - self.start

    @property
    def task_tokens(self) -> int:
        return self.in_tokens + self.out_tokens

    @property
    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        ok = sum(1 for h in self.history if h["success"])
        return ok / len(self.history) * 100


# ======================================================================
# Elm/TEA 架构 — AppState + Msg + update
# ======================================================================

from enum import Enum, auto


class MsgType(Enum):
    """TUI 消息类型——所有状态变更的触发源。"""
    USER_INPUT = auto()        # 用户输入文本
    SLASH_COMMAND = auto()     # 斜杠命令
    TASK_START = auto()        # 任务开始执行
    TASK_COMPLETE = auto()     # 任务执行完成
    TASK_ERROR = auto()        # 任务执行出错
    MODE_CHANGE = auto()       # 模式切换
    STREAM_TOGGLE = auto()     # 流式开关
    CLEAR_MESSAGES = auto()    # 清空对话
    EXIT = auto()              # 退出
    PERM_CHANGE = auto()       # 权限变更
    PERM_SET_FOLDER = auto()   # 设置文件夹路径


@dataclass
class Msg:
    """TUI 消息——Elm/TEA 中的 Msg。"""
    type: MsgType
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    """TUI 唯一状态对象——Elm/TEA 中的 Model。

    所有 TUI 状态集中在此 dataclass，状态变更只通过 update() 函数。
    """
    # 运行控制
    running: bool = True
    mode: str = "langchain"    # only langchain is exposed
    stream: bool = True         # 流式输出开关
    # 对话历史
    messages: List[Dict[str, str]] = field(default_factory=list)
    # 统计
    stats: Stats = field(default_factory=Stats)
    # 当前任务状态
    task_in_progress: bool = False
    last_result: Optional[str] = None
    last_error: Optional[str] = None
    # 累计指标
    total_tasks: int = 0
    total_tokens: int = 0
    # 文件夹权限
    folder_perm: FolderPermission = field(default_factory=FolderPermission)

    def copy(self) -> "AppState":
        """创建深拷贝（用于 update 中的不可变模式）。"""
        return copy.deepcopy(self)


def update(msg: Msg, state: AppState) -> AppState:
    """Elm/TEA update 函数——纯函数，(Msg, State) -> State。

    所有状态变更集中在此函数中，便于追踪和测试。
    """
    new_state = state.copy()

    if msg.type == MsgType.USER_INPUT:
        now = datetime.now().strftime("%H:%M:%S")
        new_state.messages.append(
            {"role": "user", "content": msg.payload.get("text", ""), "time": now})

    elif msg.type == MsgType.TASK_START:
        new_state.task_in_progress = True
        new_state.last_error = None
        new_state.stats.begin()

    elif msg.type == MsgType.TASK_COMPLETE:
        result = msg.payload.get("result", "")
        new_state.task_in_progress = False
        new_state.last_result = result
        new_state.last_error = None
        new_state.total_tasks += 1
        new_state.total_tokens += new_state.stats.task_tokens
        now = datetime.now().strftime("%H:%M:%S")
        new_state.messages.append(
            {"role": "assistant", "content": result, "time": now})
        new_state.stats.finish(success=True)

    elif msg.type == MsgType.TASK_ERROR:
        new_state.task_in_progress = False
        new_state.last_error = msg.payload.get("error", "未知错误")
        new_state.total_tasks += 1
        new_state.stats.finish(success=False)

    elif msg.type == MsgType.MODE_CHANGE:
        new_state.mode = msg.payload.get("mode", "langchain")

    elif msg.type == MsgType.STREAM_TOGGLE:
        new_state.stream = not new_state.stream

    elif msg.type == MsgType.CLEAR_MESSAGES:
        new_state.messages.clear()

    elif msg.type == MsgType.EXIT:
        new_state.running = False

    elif msg.type == MsgType.PERM_CHANGE:
        new_state.folder_perm.level = msg.payload.get("level", PermissionLevel.NONE)

    elif msg.type == MsgType.PERM_SET_FOLDER:
        new_state.folder_perm.path = msg.payload.get("path", "")

    return new_state


# ======================================================================
# CEO 驱动引擎
# ======================================================================

class CEOEngine:
    """LoopRecursive-CEO 驱动引擎。

    Phase A：FirstPrinciplesAnalyzer 递归 Top-K 决策
    Phase B：基于蓝图的无限步长 ReAct 循环
    """

    def __init__(self, llm, agent, console: Console, stats: Stats,
                 mode: str = "langchain", stream: bool = True,
                 perm_checker=None):
        self.llm = llm
        self.agent = agent
        self.console = console
        self.stats = stats
        self.mode = mode
        self.stream = stream
        self.perm_checker = perm_checker  # 文件写入权限检查回调
        self.messages: List[Dict[str, str]] = []
        self.blueprint: Optional[dict] = None
        self._error_history: List[str] = []

    # ------------------------------------------------------------------
    # Phase A: 递归第一性原理分析
    # ------------------------------------------------------------------

    def run_phase_a(self, task: str) -> dict:
        """CEO Phase A：递归 Top-K 决策分析。

        实时显示每个步骤，返回 v1 蓝图。
        """
        self.console.print(Panel(
            f"[bold yellow]Phase A：递归第一性原理分析[/bold yellow]\n"
            f"[dim]目标: {task}[/dim]",
            border_style="yellow",
        ))

        # 1. 提取关键词
        self.console.print("\n[bold]1. 提取需求关键词...[/bold]")
        try:
            from aerospace_agent.mcp.loop.recursion import FirstPrinciplesAnalyzer
            analyzer = FirstPrinciplesAnalyzer(llm=self.llm)
            keywords = analyzer._extract_keywords(task, [], {})
        except Exception:
            keywords = task.split()

        if keywords:
            for kw in keywords:
                self.console.print(f"   [cyan]• {kw}[/cyan]")
        else:
            self.console.print("   [dim](未提取到关键词)[/dim]")

        # 2. Top-K 评分
        self.console.print("\n[bold]2. Top-K 评分与排序...[/bold]")
        try:
            scored = analyzer._score_keywords(keywords, task, {})
            top_k = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
        except Exception:
            top_k = [(kw, 5.0, "", "") for kw in keywords[:5]]

        for i, (kw, score, sub_a, sub_b) in enumerate(top_k, 1):
            self.console.print(
                f"   [yellow]#{i}[/yellow] [bold]{kw}[/bold] "
                f"[dim](score: {score:.1f})[/dim]"
            )
            if sub_a:
                self.console.print(f"       [dim]├─ {sub_a}[/dim]")
            if sub_b:
                self.console.print(f"       [dim]└─ {sub_b}[/dim]")

        # 3. 递归下钻
        self.console.print("\n[bold]3. 递归下钻到第一性原理...[/bold]")
        try:
            result = analyzer.analyze(task)
            blueprint = result.get("blueprint", {})
        except Exception as e:
            self.console.print(f"   [red]分析异常: {e}[/red]")
            blueprint = {"architecture": "ReAct + tools", "error": str(e)}

        # 4. 显示蓝图
        self.console.print("\n[bold]4. v1 蓝图综合...[/bold]")
        if blueprint:
            for key, val in blueprint.items():
                if isinstance(val, str):
                    self.console.print(
                        f"   [green]✓ {key}[/green]: {val[:80]}"
                    )
                elif isinstance(val, list):
                    self.console.print(
                        f"   [green]✓ {key}[/green]: {len(val)} 项"
                    )
                elif isinstance(val, dict):
                    self.console.print(
                        f"   [green]✓ {key}[/green]: {len(val)} 个决策"
                    )

        self.blueprint = blueprint
        self.console.print(Panel(
            "[green]Phase A 完成 — 蓝图已生成[/green]",
            border_style="green",
        ))
        return blueprint

    # ------------------------------------------------------------------
    # Phase B: 受控步长 ReAct 循环
    # ------------------------------------------------------------------

    def run_phase_b(self, task: str) -> str:
        """CEO Phase B：基于蓝图的受控步长循环。

        委托 AerospaceAgent.run_react_stream() 执行,集成上下文管理+记忆+智能工具执行。
        保留流式输出与 Ctrl+C 中断能力。
        """
        self.console.print(Panel(
            f"[bold blue]Phase B：受控步长执行循环[/bold blue]\n"
            f"[dim]委托 Agent 统一 ReAct 入口,集成上下文/记忆/蓝图[/dim]",
            border_style="blue",
        ))

        # 流式回调 — 逐字实时输出到终端
        chunks_collected = []

        def _stream_cb(chunk: str):
            try:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                chunks_collected.append(chunk)
                self.stats.out_tokens += len(chunk) // 4
            except Exception:
                pass

        # 委托 Agent 执行统一 ReAct 循环
        try:
            max_steps = int(os.environ.get("AEROSPACE_REACT_MAX_STEPS", "20"))
            result = self.agent.run_react_stream(
                task=task,
                blueprint=self.blueprint,
                max_steps=max_steps,
                stream_callback=_stream_cb if self.stream else None,
                enable_context=True,
            )
        except KeyboardInterrupt:
            sys.stdout.write("\n[yellow]⚠ 用户中断 — 输出已停止[/yellow]\n")
            sys.stdout.flush()
            result = "".join(chunks_collected) or "用户中断"
            return result

        # 换行结束流式输出
        if self.stream:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.stats.llm_call(task, result)

        self.console.print(Panel(
            f"[bold green]{result[:500]}[/bold green]",
            title="[bold green]✓ 初版完成[/bold green]",
            border_style="green",
        ))
        return result

    # ------------------------------------------------------------------
    # 简单聊天模式（无 ReAct，纯流式对话）
    # ------------------------------------------------------------------

    def run_chat(self, task: str) -> str:
        """纯聊天模式：流式输出，无 CEO 分析，无 ReAct。"""
        self.console.print("[green]💬 Chat:[/green] ", end="")
        sys.stdout.flush()

        messages = [
            {"role": "system", "content": "你是航天导航控制助手。简洁回答。"},
            {"role": "user", "content": task},
        ]

        response = ""
        try:
            if self.stream and hasattr(self.llm, "stream_chat"):
                for chunk in self.llm.stream_chat(messages, max_tokens=2000):
                    response += chunk
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                response = self.llm.chat(messages)
                self.console.print(response)

            self.stats.llm_call(task, response)
            self.stats.llm_calls = 1
        except KeyboardInterrupt:
            sys.stdout.write("\n[yellow]⚠ 用户中断[/yellow]\n")
            sys.stdout.flush()
            return response or "(已中断)"
        except Exception as e:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.console.print(f"[red]LLM 错误: {e}[/red]")
            return f"错误: {e}"

        return response

    # ------------------------------------------------------------------
    # 完整执行
    # ------------------------------------------------------------------

    def execute(self, task: str) -> str:
        """根据当前模式执行任务。"""
        self.stats.begin()
        stream_callback = (
            (lambda chunk: sys.stdout.write(chunk)) if self.stream else None
        )
        result = self.agent.run_langchain(
            task,
            stream_callback=stream_callback,
        )

        success = bool(result) and "失败" not in result
        self.stats.finish(success)
        return result


# ======================================================================
# CLI 终端
# ======================================================================

class CLITerminal:
    """CodeWhale 风格 CLI 终端。"""

    def __init__(self, use_mock: bool = False, use_local: bool = False):
        self.console = Console()
        self.use_mock = use_mock
        self.llm = None
        self.llm_name = "未连接"
        self.agent = None
        # --local 标志：设置本地 LLM 基址环境变量（若尚未设置）
        if use_local and not os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL"):
            os.environ["AEROSPACE_LOCAL_LLM_BASE_URL"] = QWEN3_URL
        # Elm/TEA: 唯一状态对象
        self.state = AppState(stats=Stats())
        self._connect()
        self._init_agent()
        self._init_folder_permission()

    # ------------------------------------------------------------------
    # Elm/TEA: 属性委托——现有渲染代码通过 self.messages 等访问状态
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.state.running

    @property
    def mode(self) -> str:
        return self.state.mode

    @property
    def stream(self) -> bool:
        return self.state.stream

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self.state.messages

    @property
    def stats(self) -> Stats:
        return self.state.stats

    def _connect(self):
        """连接 LLM。"""
        if self.use_mock:
            from aerospace_agent.core.llm_interface import MockLLM
            self.llm = MockLLM()
            self.llm_name = "MockLLM(离线)"
            self.console.print("[cyan]使用 MockLLM[/cyan]")
            return
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"{QWEN3_URL}/models", timeout=5
            ) as resp:
                data = json.loads(resp.read().decode())
                model_id = (
                    data["data"][0]["id"]
                    if data.get("data") else QWEN3_MODEL
                )
            from aerospace_agent.core.llm_interface import LocalLLM
            self.llm = LocalLLM(base_url=QWEN3_URL, model=model_id)
            self.llm_name = f"Qwen3({model_id})"
            self.console.print(
                f"[green]✓ Qwen3 已连接[/green] [dim]({QWEN3_URL})[/dim]"
            )
        except Exception as e:
            from aerospace_agent.core.llm_interface import MockLLM
            self.llm = MockLLM()
            self.llm_name = "MockLLM(回退)"
            self.console.print(
                f"[yellow]⚠ Qwen3 不可达: {e}，回退 MockLLM[/yellow]"
            )

    def _init_agent(self):
        try:
            from aerospace_agent.core.agent import create_default_agent
            self.agent = create_default_agent(
                max_steps=999, force_mock=False
            )
            self.agent.llm = self.llm
            n_native = len(self.agent.tools)
            n_mcp = len(self.agent.mcp_tools)
            self.console.print(
                f"[dim]  工具: {n_native}+{n_mcp}  "
                f"步数: 无限制[/dim]"
            )
        except Exception as e:
            self.console.print(f"[red]Agent 初始化失败: {e}[/red]")

    def _init_folder_permission(self):
        """启动时询问当前文件夹的编辑权限。"""
        cwd = os.getcwd()
        self.state = update(
            Msg(MsgType.PERM_SET_FOLDER, {"path": cwd}), self.state)

        # 扫描文件夹内容
        try:
            entries = os.listdir(cwd)
            py_files = [f for f in entries if f.endswith(".py")]
            subdirs = [d for d in entries if os.path.isdir(os.path.join(cwd, d))]
            n_files = len(entries)
        except Exception:
            n_files = 0
            py_files = []
            subdirs = []

        perm = self.state.folder_perm
        self.console.print(Panel(
            f"[bold cyan]文件夹权限请求[/bold cyan]\n\n"
            f"[dim]路径:[/dim] [white]{cwd}[/white]\n"
            f"[dim]内容:[/dim] {n_files} 项"
            f"（{len(py_files)} Python 文件, {len(subdirs)} 子目录）\n\n"
            f"[bold yellow]Agent 需要编辑此文件夹的权限才能执行文件操作。[/bold yellow]\n"
            f"请选择权限级别：\n"
            f"  [1] {PermissionLevel.LABELS[PermissionLevel.ASK]}"
            f" [yellow]— 每次编辑前询问确认[/yellow]\n"
            f"  [2] {PermissionLevel.LABELS[PermissionLevel.AUTO]}"
            f" [green]— 全自动审批，无需确认[/green]\n"
            f"  [3] {PermissionLevel.LABELS[PermissionLevel.NONE]}"
            f" [red]— 禁止编辑文件[/red]\n"
            f"  [Enter] 默认: {PermissionLevel.LABELS[PermissionLevel.ASK]}",
            border_style="yellow",
            title="[bold yellow]权限设置[/bold yellow]",
        ))

        try:
            choice = input("选择 [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        level_map = {
            "1": PermissionLevel.ASK,
            "2": PermissionLevel.AUTO,
            "3": PermissionLevel.NONE,
            "": PermissionLevel.ASK,
        }
        level = level_map.get(choice, PermissionLevel.ASK)
        self.state = update(
            Msg(MsgType.PERM_CHANGE, {"level": level}), self.state)

        color = PermissionLevel.COLORS[level]
        label = PermissionLevel.LABELS[level]
        self.console.print(
            f"[{color}]✓ 权限设置: {label}[/{color}] "
            f"[dim]({cwd})[/dim]"
        )

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render_header(self):
        """顶部状态栏。"""
        perm = self.state.folder_perm
        perm_color = PermissionLevel.COLORS.get(perm.level, "red")
        perm_label = PermissionLevel.LABELS.get(perm.level, "未知")
        self.console.print(Rule(style="blue"))
        self.console.print(
            f"[bold cyan]aerospace-agent[/bold cyan] v{VERSION}  "
            f"[dim]|[/dim]  [blue]{self.llm_name}[/blue]  "
            f"[dim]|[/dim]  [yellow]{self.mode}[/yellow]"
            f"[dim]/{'stream' if self.stream else 'nostream'}[/dim]  "
            f"[dim]|[/dim]  [{perm_color}]{perm_label}[/{perm_color}]  "
            f"[dim]|[/dim]  [white]{os.getcwd()}[/white]"
        )
        self.console.print(Rule(style="blue"))

    def render_footer(self):
        """底部统计栏。"""
        s = self.stats
        self.console.print(Rule(style="dim"))
        self.console.print(
            f"[dim]Token:[/dim] [green]{s.task_tokens:,}[/green]"
            f"[dim](in:{s.in_tokens:,} out:{s.out_tokens:,})[/dim]  "
            f"[dim]耗时:[/dim] [yellow]{s.elapsed:.1f}s[/yellow]  "
            f"[dim]LLM:[/dim] {s.llm_calls}  "
            f"[dim]工具:[/dim] {s.tool_calls}  "
            f"[dim]步数:[/dim] {s.react_steps}  "
            f"[dim]累计:[/dim] [green]{s.total_tokens:,}[/green]tokens "
            f"[yellow]{s.total_time:.1f}s[/yellow]  "
            f"[dim]成功率:[/dim] [green]{s.success_rate:.0f}%[/green]"
        )

    def render_transcript(self):
        """对话历史。"""
        if not self.messages:
            return
        for msg in self.messages[-10:]:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("time", "")
            if role == "user":
                self.console.print(Panel(
                    Text(content, style="cyan"),
                    title=f"[bold cyan]你[/bold cyan] {ts}",
                    border_style="cyan",
                    padding=(0, 1),
                ))
            elif role == "assistant":
                self.console.print(Panel(
                    Text(content, style="green"),
                    title=f"[bold green]Agent[/bold green] {ts}",
                    border_style="green",
                    padding=(0, 1),
                ))

    # ------------------------------------------------------------------
    # 主循环 — Elm/TEA: init → (update → view) loop
    # ------------------------------------------------------------------

    def run(self):
        """主交互循环——Elm/TEA 模式。

        loop:  msg = get_input() → state = update(msg, state) → view(state)
        所有状态变更通过 update() 函数，便于追踪和测试。
        """
        mode_desc = {
            "langchain": "LangChain 基础模式（滑动窗口记忆 + RAG/MCP 接口）",
        }
        self._view_header_banner(mode_desc)

        while self.state.running:
            # Composer — 获取输入
            self.console.print(Rule(style="blue"))
            try:
                user_input = input("aerospace> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.state = update(Msg(MsgType.EXIT), self.state)
                self.console.print("\n[magenta]再见！[/magenta]")
                break

            if not user_input:
                continue

            # 斜杠命令 → dispatch Msg
            if user_input.startswith("/"):
                self._handle_slash_tea(user_input)
                continue

            # 用户输入 → Msg(USER_INPUT)
            self.state = update(
                Msg(MsgType.USER_INPUT, {"text": user_input}), self.state)

            # 任务开始 → Msg(TASK_START)
            self.state = update(Msg(MsgType.TASK_START), self.state)
            self.render_header()

            # 执行任务
            try:
                ceo = CEOEngine(
                    self.llm, self.agent, self.console, self.state.stats,
                    mode=self.state.mode, stream=self.state.stream,
                    perm_checker=self.check_file_write,
                )
                result = ceo.execute(user_input)
                # 任务完成 → Msg(TASK_COMPLETE)
                self.state = update(
                    Msg(MsgType.TASK_COMPLETE, {"result": result or ""}),
                    self.state)
            except Exception as exc:
                # 任务出错 → Msg(TASK_ERROR)
                self.state = update(
                    Msg(MsgType.TASK_ERROR, {"error": str(exc)}), self.state)
                self.console.print(
                    f"[red]任务出错: {exc}[/red]")

            self.render_footer()

        # 退出统计
        self._view_exit_summary()

    def _view_header_banner(self, mode_desc: Dict[str, str]):
        """渲染启动横幅。"""
        self.console.print(Panel(
            f"[bold blue]aerospace-agent CLI v{VERSION}[/bold blue]\n"
            f"[bold yellow]{mode_desc.get(self.state.mode, self.state.mode)}[/bold yellow]\n"
            f"模型: [green]{self.llm_name}[/green]  "
            f"流式: [{'green' if self.state.stream else 'red'}]{'开' if self.state.stream else '关'}[/{'green' if self.state.stream else 'red'}]\n"
            f"输入任务开始，或 /help 查看命令",
            border_style="blue",
        ))

    def _view_exit_summary(self):
        """渲染退出统计。"""
        self.console.print(
            f"\n[magenta]再见！累计 "
            f"{self.state.stats.total_tokens:,} tokens, "
            f"{self.state.stats.total_time:.1f}s, "
            f"{len(self.state.stats.history)} 任务[/magenta]"
        )

    def _handle_slash_tea(self, cmd: str):
        """斜杠命令——Elm/TEA 模式，通过 Msg 更新状态。"""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()

        if command == "/help":
            self._view_help()

        elif command == "/mode":
            if len(parts) > 1:
                m = parts[1].strip().lower()
                if m == "langchain":
                    self.state = update(
                        Msg(MsgType.MODE_CHANGE, {"mode": m}), self.state)
                    self.console.print(
                        "[green]OK 当前仅保留 LangChain 基础模式[/green]")
                else:
                    self.console.print(
                        f"[red]未知模式: {m}（当前仅支持: langchain）[/red]")
            else:
                self.console.print(
                    f"[cyan]当前模式: {self.state.mode}[/cyan]\n"
                    f"[dim]当前仅支持: langchain[/dim]")

        elif command == "/stream":
            self.state = update(Msg(MsgType.STREAM_TOGGLE), self.state)
            self.console.print(
                f"[green]流式输出: {'开' if self.state.stream else '关'}[/green]")

        elif command == "/perm":
            self._handle_perm_command(parts[1] if len(parts) > 1 else "")

        elif command == "/tools":
            self._view_tools()

        elif command == "/stats":
            self._view_stats()

        elif command == "/clear":
            self.state = update(Msg(MsgType.CLEAR_MESSAGES), self.state)
            self.console.print("[green]✓ 对话已清空[/green]")

        elif command in ("/exit", "/quit"):
            self.state = update(Msg(MsgType.EXIT), self.state)

        else:
            self.console.print(f"[red]未知命令: {command}[/red]")

    def _view_help(self):
        """渲染帮助表格。"""
        t = Table(title="命令", show_header=True, box=box.ROUNDED)
        t.add_column("命令", style="cyan", width=12)
        t.add_column("说明", style="white")
        for c, d in [("/help", "显示帮助"), ("/mode", "查看当前模式"),
                     ("/stream", "切换流式输出 (开/关)"),
                     ("/perm", "文件夹权限 (ask/auto/none)"),
                     ("/tools", "工具列表"),
                     ("/stats", "统计"), ("/clear", "清空对话"),
                     ("/exit", "退出")]:
            t.add_row(c, d)
        self.console.print(t)

    # ------------------------------------------------------------------
    # 权限管理
    # ------------------------------------------------------------------

    def _handle_perm_command(self, arg: str):
        """处理 /perm 命令。"""
        perm = self.state.folder_perm

        if not arg:
            # 显示当前权限
            color = PermissionLevel.COLORS.get(perm.level, "red")
            label = PermissionLevel.LABELS.get(perm.level, "未知")
            self.console.print(Panel(
                f"[dim]路径:[/dim] [white]{perm.path or os.getcwd()}[/white]\n"
                f"[dim]权限:[/dim] [{color}]{label}[/{color}]\n\n"
                f"[dim]切换: /perm ask | /perm auto | /perm none[/dim]",
                title="[bold yellow]文件夹权限[/bold yellow]",
                border_style="yellow",
            ))
            return

        level = arg.strip().lower()
        if level not in PermissionLevel.ALL:
            self.console.print(
                f"[red]未知权限级别: {level}[/red]\n"
                f"[dim]可选: ask / auto / none[/dim]"
            )
            return

        self.state = update(
            Msg(MsgType.PERM_CHANGE, {"level": level}), self.state)
        color = PermissionLevel.COLORS[level]
        label = PermissionLevel.LABELS[level]
        self.console.print(
            f"[{color}]✓ 权限切换: {label}[/{color}]"
        )

    def check_file_write(self, file_path: str) -> bool:
        """检查是否允许写入文件 — 供 CEOEngine 调用。

        根据权限级别：
        - none: 拒绝
        - ask:  弹出确认提示
        - auto: 直接允许
        """
        perm = self.state.folder_perm
        if not perm.is_writable():
            self.console.print(
                f"[red]✗ 权限拒绝: 禁止编辑文件 ({file_path})[/red]\n"
                f"[dim]使用 /perm ask 或 /perm auto 开启权限[/dim]"
            )
            return False

        if perm.should_ask():
            self.console.print(
                f"[yellow]⚠ 文件写入请求: {file_path}[/yellow]"
            )
            try:
                ans = input("  允许写入? [y/N]: ").strip().lower()
                if ans not in ("y", "yes", "是"):
                    self.console.print("[red]已拒绝[/red]")
                    return False
            except (EOFError, KeyboardInterrupt):
                return False

        return True

    def _view_tools(self):
        """渲染工具列表。"""
        if not self.agent:
            return
        t = Table(title="工具", show_header=True, box=box.ROUNDED)
        t.add_column("名称", style="cyan", width=18)
        t.add_column("类型", style="blue", width=8)
        t.add_column("说明", style="white")
        for n, tool in self.agent.tools.items():
            t.add_row(n, "native", tool.description[:40])
        for n, bt in self.agent.mcp_tools.items():
            t.add_row(n, "mcp",
                      getattr(bt, "description", "")[:40])
        self.console.print(t)

    def _view_stats(self):
        """渲染统计表格。"""
        s = self.state.stats
        t = Table(title="统计", show_header=True, box=box.ROUNDED)
        t.add_column("指标", style="cyan", width=12)
        t.add_column("累计", style="white")
        t.add_row("Token", f"{s.total_tokens:,}")
        t.add_row("耗时", f"{s.total_time:.1f}s")
        t.add_row("任务数", str(len(s.history)))
        t.add_row("成功率", f"{s.success_rate:.0f}%")
        if s.history:
            for i, h in enumerate(s.history[-5:], 1):
                st = "✓" if h["success"] else "✗"
                t.add_row(
                    f"  #{i}", f"{st} {h['tokens']:,}t {h['time']}s"
                )
        self.console.print(t)


def main():
    import argparse
    p = argparse.ArgumentParser(description="aerospace-agent CLI")
    p.add_argument("--mock", action="store_true",
                   help="使用 MockLLM（离线模式）")
    args = p.parse_args()
    CLITerminal(use_mock=args.mock).run()


if __name__ == "__main__":
    main()
