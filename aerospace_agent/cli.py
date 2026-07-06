"""航天导航控制 Agent 命令行接口。

基于 Click，提供以下子命令：

    run <task>            执行一个任务（调用 AerospaceAgent.run）
    workflow list         列出可用工作流
    workflow run <name>   运行指定工作流（--param key=value 可多次）
    tool list             列出可用工具及其可用性
    rag query <query>     检索知识库
    rag index <dir>       索引文档目录
    demo                  运行地月转移轨道演示（lazy import）
    git <subcommand>      git 操作包装（init/add/commit/status/log/branch/checkout）
    shell                 进入交互式 REPL

彩色输出优先使用 rich，未安装时自动回退到 print。
"""
from __future__ import annotations

import sys

try:
    import click
except ImportError:  # pragma: no cover
    print("需要安装 click：pip install click")
    sys.exit(1)

# ----------------------------------------------------------------------
# 彩色输出（rich 可选，未装则回退 print）
# ----------------------------------------------------------------------
try:
    from rich.console import Console

    _console = Console()

    def cprint(msg, style=None):
        """带样式的输出（rich 可用时）。"""
        if style:
            _console.print(msg, style=style)
        else:
            _console.print(msg)
except Exception:  # pragma: no cover
    def cprint(msg, style=None):
        print(msg)


# 全局输出级别（由 --verbose / --quiet 控制 CLI 横幅日志）
VERBOSE = True


def log(msg, style=None):
    """根据 VERBOSE 级别输出 CLI 日志。"""
    if VERBOSE:
        cprint(msg, style=style)


# ----------------------------------------------------------------------
# 主命令组
# ----------------------------------------------------------------------
@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细输出")
@click.option("--quiet", "-q", is_flag=True, default=False, help="静默输出")
@click.version_option(version="0.1.0", prog_name="aerospace-agent")
@click.pass_context
def cli(ctx, verbose, quiet):
    """航天导航控制 Agent 命令行工具。"""
    global VERBOSE
    if quiet:
        VERBOSE = False
    else:
        VERBOSE = True  # 默认输出；--verbose 保留以兼容
    ctx.ensure_object(dict)


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
@cli.command()
@click.argument("task")
@click.option("--mock", is_flag=True, default=False, help="强制使用 MockLLM")
@click.option("--steps", default=10, show_default=True, help="最大 ReAct 步数")
def run(task, mock, steps):
    """执行一个任务。"""
    # lazy import 避免启动开销与潜在循环
    from .core.agent import create_default_agent

    agent = create_default_agent(max_steps=steps, force_mock=mock)
    result = agent.run(task)
    cprint("\n===== 执行结果 =====", style="bold green")
    cprint(result, style="green")


# ----------------------------------------------------------------------
# workflow 组
# ----------------------------------------------------------------------
@cli.group()
def workflow():
    """工作流管理。"""
    pass


@workflow.command("list")
def workflow_list():
    """列出可用工作流。"""
    from .core.agent import create_default_agent

    agent = create_default_agent(force_mock=True)
    cprint("可用工作流：", style="bold cyan")
    if not agent.workflows:
        cprint("  (无)", style="yellow")
    for name, wf in agent.workflows.items():
        cprint(f"  - {name}: {wf.description}", style="cyan")


@workflow.command("run")
@click.argument("name")
@click.option("--param", "-p", multiple=True, help="参数 key=value，可多次指定")
def workflow_run(name, param):
    """运行指定工作流。"""
    from .core.agent import create_default_agent

    agent = create_default_agent(force_mock=True)
    wf = agent.workflows.get(name)
    if not wf:
        cprint(f"未找到工作流: {name}", style="bold red")
        sys.exit(1)
    # 解析参数
    params = {}
    for p in param:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
        else:
            cprint(f"忽略无效参数（应为 key=value）: {p}", style="yellow")
    cprint(f"运行工作流: {name}", style="bold cyan")
    results = wf.run(agent, **params)
    for r in results:
        cprint(str(r), style="green")


# ----------------------------------------------------------------------
# tool 组
# ----------------------------------------------------------------------
@cli.group()
def tool():
    """工具管理。"""
    pass


@tool.command("list")
def tool_list():
    """列出可用工具及其可用性。"""
    from .core.agent import create_default_agent

    agent = create_default_agent(force_mock=True)
    cprint("可用工具：", style="bold cyan")
    if not agent.tools and not agent.mcp_tools:
        cprint("  (无)", style="yellow")
    # 原生工具
    for name, t in agent.tools.items():
        cprint(f"  - {name}: {t.description} [可用]", style="cyan")
    # MCP 工具（含可用性检测：真实库可用 / 回退模式可用）
    for name, bt in agent.mcp_tools.items():
        source = getattr(bt, "source", "unknown")
        if source == "real":
            label, style = "真实库可用", "green"
        elif source == "fallback":
            label, style = "回退模式可用", "yellow"
        else:
            label, style = f"source={source}", "yellow"
        desc = getattr(bt, "description", "")
        cprint(f"  - {name}: {desc} [{label}]", style=style)


# ----------------------------------------------------------------------
# rag 组
# ----------------------------------------------------------------------
@cli.group()
def rag():
    """知识库检索。"""
    pass


@rag.command("query")
@click.argument("query")
@click.option("--top-k", default=3, show_default=True, help="返回条数")
def rag_query(query, top_k):
    """检索知识库。"""
    from .core.agent import create_default_agent

    agent = create_default_agent(force_mock=True)
    rag_obj = getattr(agent, "rag", None)
    if rag_obj is None:
        cprint("RAG 不可用", style="red")
        return
    results = rag_obj.query(query, top_k=top_k)
    if not results:
        cprint("未检索到相关内容（知识库可能为空，请先执行 rag index）", style="yellow")
        return
    cprint(f"检索结果（top {top_k}）：", style="bold cyan")
    for r in results:
        cprint(r, style="green")
        cprint("---", style="dim")


@rag.command("index")
@click.argument("dir")
def rag_index(dir):
    """索引文档目录。"""
    from .core.agent import create_default_agent

    agent = create_default_agent(force_mock=True)
    rag_obj = getattr(agent, "rag", None)
    if rag_obj is None:
        cprint("RAG 不可用", style="red")
        return
    count = rag_obj.index(dir)
    cprint(f"已索引 {count} 个段落（来自 {dir}）", style="bold green")


@rag.command("literature")
@click.argument("query")
@click.option("--topic", "-t", default="", help="研究主题（相关性评分用，默认等同 query）")
@click.option("--max", "max_results", default=10, show_default=True, help="最大搜索数")
@click.option("--no-download", is_flag=True, default=False, help="仅评分不下载 PDF")
@click.option("--report", is_flag=True, default=False, help="同时生成知识学习报告")
def rag_literature(query, topic, max_results, no_download, report):
    """搜索最新文献、评估相关性、下载强相关论文并总结。

    流程：CSTCloud 登录 → arXiv 搜索 → 摘要评分 →
    strong 相关：下载 PDF + 全文总结 + 索引入 RAG + 更新知识图谱 →
    weak 相关：跳过 → 生成动态知识云图。

    示例：
      aerospace-agent rag literature "lunar transfer orbit"
      aerospace-agent rag literature "spacecraft trajectory optimization" --topic "地月转移轨道" --report
    """
    from .rag.aerospace_rag import AerospaceRAG

    rag_obj = AerospaceRAG()
    cprint("=== 文献搜索与处理管线 ===", style="bold magenta")
    cprint(f"搜索查询: {query}", style="cyan")
    cprint(f"研究主题: {topic or query}", style="cyan")
    cprint(f"最大结果数: {max_results}", style="cyan")
    cprint(f"下载强相关: {'否' if no_download else '是'}", style="cyan")
    cprint("", style="")

    result = rag_obj.search_literature(
        query=query,
        research_topic=topic,
        max_results=max_results,
        download_strong=not no_download,
    )

    cprint("\n=== 管线结果 ===", style="bold green")
    cprint(f"搜索到: {result['total_found']} 篇", style="cyan")
    cprint(f"强相关: {result['strong_count']} 篇", style="green")
    cprint(f"弱相关: {result['weak_count']} 篇（已跳过）", style="yellow")
    cprint(f"已下载: {result['downloaded_count']} 篇", style="green")

    cprint("\n--- 论文明细 ---", style="bold cyan")
    for i, p in enumerate(result.get("papers", []), 1):
        rel_style = "green" if p["relevance"] == "strong" else "yellow"
        status_icon = {"downloaded": "[已下载]", "skipped": "[跳过]", "failed": "[失败]"}.get(p["status"], "[?]")
        cprint(f"\n{i}. {p['title']}", style="bold white")
        cprint(f"   arXiv: {p['arxiv_id']}  作者: {', '.join(p['authors'])}", style="dim")
        cprint(f"   相关性: {p['relevance']} (score={p['score']:.2f})  {status_icon}", style=rel_style)
        if p["summary"]:
            cprint(f"   总结: {p['summary']}", style="dim")
        if p["pdf_path"]:
            cprint(f"   PDF: {p['pdf_path']}", style="dim")
        if p["concepts"]:
            cprint(f"   概念: {', '.join(p['concepts'])}", style="dim")

    # 生成知识云图
    cprint("\n=== 生成动态知识云图 ===", style="bold magenta")
    cloud_path = rag_obj.generate_knowledge_cloud()
    cprint(f"知识云图: {cloud_path}", style="green")

    # 可选：生成知识学习报告
    if report:
        cprint("\n=== 生成知识学习报告 ===", style="bold magenta")
        report_path = rag_obj.generate_knowledge_report()
        cprint(f"学习报告: {report_path}", style="green")

    # 知识图谱快照
    snap = result.get("knowledge_graph_snapshot", {})
    if snap:
        cprint(f"\n知识图谱: {snap.get('num_nodes', snap.get('nodes', '?'))} 节点, "
               f"{snap.get('num_edges', snap.get('edges', '?'))} 边", style="cyan")


@rag.command("cloud")
@click.option("--output", "-o", default="/workspace/reports/knowledge_cloud.html", show_default=True, help="输出路径")
def rag_cloud(output):
    """生成动态知识云图（力导向交互式 HTML）。"""
    from .rag.aerospace_rag import AerospaceRAG

    rag_obj = AerospaceRAG()
    path = rag_obj.generate_knowledge_cloud(output_path=output)
    s = rag_obj.status()
    cprint(f"知识云图已生成: {path}", style="bold green")
    cprint(f"节点数: {s.get('kg_nodes', '?')}  边数: {s.get('kg_edges', '?')}  "
           f"已下载文献: {s.get('downloaded_papers', 0)}", style="cyan")


@rag.command("report")
@click.option("--output", "-o", default="/workspace/reports/knowledge_learning_report.html", show_default=True, help="输出路径")
def rag_report(output):
    """生成知识学习报告（概念网络分析 + 论文写作辅助）。"""
    from .rag.aerospace_rag import AerospaceRAG

    rag_obj = AerospaceRAG()
    path = rag_obj.generate_knowledge_report(output_path=output)
    cprint(f"知识学习报告已生成: {path}", style="bold green")


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------
@cli.command()
@click.option("--task", default="设计地月转移轨道", show_default=True, help="演示任务")
@click.option("--full/--react-only", default=True, show_default=True,
              help="--full 运行完整端到端链路（工作流+图+报告）；--react-only 仅 Agent ReAct")
def demo(task, full):
    """运行地月转移轨道演示。

    --full（默认）：Agent ReAct + TrajectoryAnalysisWorkflow + Basilisk 可视化
                   + 7 张分析图 + 自包含 HTML 报告。
    --react-only  ：仅运行 Agent ReAct 循环。
    """
    cprint("=== 航天导航控制 Agent 演示 ===", style="bold magenta")
    cprint(f"演示任务: {task}", style="magenta")
    if full:
        from .demo import run_full_demo
        artifacts = run_full_demo(task)
        cprint("\n===== 演示产物 =====", style="bold green")
        for k, v in artifacts.items():
            cprint(f"  {k}: {v}", style="green")
    else:
        from .core.agent import create_default_agent
        agent = create_default_agent(max_steps=10, force_mock=True)
        result = agent.run(task)
        cprint("\n===== 演示结果 =====", style="bold green")
        cprint(result, style="green")


# ----------------------------------------------------------------------
# git 组
# ----------------------------------------------------------------------
@cli.group()
def git():
    """Git 操作包装。"""
    pass


@git.command("init")
@click.option("--path", default=".", show_default=True, help="仓库路径")
def git_init(path):
    """初始化 git 仓库。"""
    from .utils.git_manager import GitManager

    gm = GitManager(path)
    cprint(gm.init(path), style="green")


@git.command("add")
@click.argument("pathspec", default=".")
def git_add(pathspec):
    """将文件加入暂存区。"""
    from .utils.git_manager import GitManager

    gm = GitManager(".")
    cprint(gm.add(pathspec) or "已添加", style="green")


@git.command("commit")
@click.argument("msg")
def git_commit(msg):
    """提交暂存区变更。"""
    from .utils.git_manager import GitManager

    gm = GitManager(".")
    cprint(gm.commit(msg), style="green")


@git.command("status")
def git_status():
    """查看工作区状态。"""
    from .utils.git_manager import GitManager

    gm = GitManager(".")
    cprint(gm.status() or "工作区干净", style="cyan")


@git.command("log")
@click.option("-n", "--number", "n", default=10, show_default=True, help="条数")
def git_log(n):
    """查看提交日志。"""
    from .utils.git_manager import GitManager

    gm = GitManager(".")
    cprint(gm.log(n) or "无提交记录", style="cyan")


@git.command("branch")
@click.argument("name")
def git_branch(name):
    """创建新分支。"""
    from .utils.git_manager import GitManager

    gm = GitManager(".")
    cprint(gm.create_branch(name), style="green")


@git.command("checkout")
@click.argument("name")
def git_checkout(name):
    """切换分支。"""
    from .utils.git_manager import GitManager

    gm = GitManager(".")
    cprint(gm.checkout(name), style="green")


# ----------------------------------------------------------------------
# shell
# ----------------------------------------------------------------------
@cli.command()
def shell():
    """进入交互式 REPL。"""
    cprint("航天导航控制 Agent 交互模式（输入 exit 退出）", style="bold magenta")
    from .core.agent import create_default_agent

    agent = create_default_agent(force_mock=True)
    while True:
        try:
            line = input("aerospace> ").strip()
        except (EOFError, KeyboardInterrupt):
            cprint("\n再见", style="magenta")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            cprint("再见", style="magenta")
            break
        if line.lower() in ("help", "?"):
            cprint("输入任务描述执行，或输入 exit 退出", style="cyan")
            continue
        try:
            result = agent.run(line)
            cprint(result, style="green")
        except Exception as e:
            cprint(f"执行出错: {e}", style="red")


def main():
    """入口函数（console_scripts 与 python -m 共用）。"""
    cli(obj={})


if __name__ == "__main__":
    main()
