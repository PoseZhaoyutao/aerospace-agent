"""上下文管理 — 1:1 复刻 CCB context.ts。

管理系统提示词的组装：default system prompt + user context + system context。

关键设计（照搬 CCB）：
    1. System prompt 分三层：default(静态) + userContext(半静态) + systemContext(半静态)
    2. userContext: 项目记忆文件（CLAUDE.md 等价物）、当前日期
    3. systemContext: git 状态等环境信息
    4. 三层分离使得 prompt caching 可生效（静态部分可缓存）
"""
from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from aerospace_agent.local_runtime import run_command


# ======================================================================
# 系统提示词常量
# ======================================================================

DEFAULT_SYSTEM_PROMPT = """你是航天动力学科研 Agent。你通过调用工具来完成任务。

## 核心工作流
1. 分析用户任务，确定需要哪些信息或操作
2. 检查可用工具列表，选择最匹配的工具
3. 调用工具执行操作
4. 检查工具返回结果是否正确
5. 如果结果有误或工具不可用，换用替代工具或调整参数重试
6. 如果现有工具无法完成任务，用 create_tool 创建新工具
7. 创建工具后立即调用它，不要只创建不使用
8. 任务完成后给出清晰的最终答案

## 工具使用规则
- 每次可以调用一个或多个工具
- 只读工具可以并发执行（如查询、计算、读取文件）
- 写操作工具串行执行（如保存文件、创建工具）
- 工具结果会以 tool_result 形式返回
- 大输出会自动写入文件，不要在回复中重复大段内容
- 如果工具结果包含 [验证提示]，说明执行有问题，需要根据提示调整

## 工具选择策略
- 数学计算：优先使用 calculator 工具
- 文件操作：save_file / read_file / list_files / search_files
- 数据解析：parse_csv / parse_json / parse_text
- 可视化：plot_line / plot_bar / plot_scatter / plot_pie / histogram_compute
- 航天专用：orbit_design / launch_window / lunar_transfer / basilisk_viz
- 文献检索：literature_review / search_papers
- 自我进化：create_tool / list_tools / tool_help
- 代码执行：python_execute

## 多步推理指引
- 复杂任务分解为多个步骤，逐步执行
- 每步执行后验证结果，确认无误再继续下一步
- 如果中间步骤失败，不要放弃，分析原因并调整策略
- 最终答案应整合所有步骤的结果，给出完整结论

## 重要约束
- 不要臆测工具返回值，以实际返回为准
- 不要在未调用工具的情况下编造结果
- 如果多次尝试仍无法完成，如实说明原因和已尝试的方法
- 工具执行完成后，必须给出最终文字回答总结结果，不要只输出工具调用
- 最终回答应该是自然语言，不要包含 [Tool Call: ...] 格式"""


# ======================================================================
# User Context — 项目记忆 + 日期
# ======================================================================

@lru_cache(maxsize=1)
def get_user_context(
    project_memory_path: Optional[str] = None,
) -> Dict[str, str]:
    """获取用户上下文 — 项目记忆文件 + 当前日期。

    对应 CCB 的 getUserContext() — 读取 CLAUDE.md 等价物。
    """
    context: Dict[str, str] = {}

    # 项目记忆文件（CLAUDE.md 等价物）
    memory_content = _load_project_memory(project_memory_path)
    if memory_content:
        context["projectMemory"] = memory_content

    # 当前日期
    context["currentDate"] = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."

    return context


def _load_project_memory(path: Optional[str] = None) -> Optional[str]:
    """加载项目记忆文件。

    查找顺序：
    1. 显式指定的路径
    2. 当前目录下的 CLAUDE.md / PROJECT_MEMORY.md / .project_memory
    3. 用户主目录下的 .aerospace/memory.md
    """
    candidates = []
    if path:
        candidates.append(path)
    cwd = os.getcwd()
    candidates.extend([
        os.path.join(cwd, "CLAUDE.md"),
        os.path.join(cwd, "PROJECT_MEMORY.md"),
        os.path.join(cwd, ".project_memory"),
    ])
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, ".aerospace", "memory.md"))

    for candidate in candidates:
        try:
            if os.path.isfile(candidate) and os.path.getsize(candidate) < 100000:
                with open(candidate, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception:
            continue
    return None


# ======================================================================
# System Context — 环境信息
# ======================================================================

@lru_cache(maxsize=1)
def get_system_context() -> Dict[str, str]:
    """获取系统上下文 — git 状态等环境信息。

    对应 CCB 的 getSystemContext()。
    """
    context: Dict[str, str] = {}

    # Git 状态
    git_status = _get_git_status()
    if git_status:
        context["gitStatus"] = git_status

    return context


def _get_git_status() -> Optional[str]:
    """获取 git 状态摘要。"""
    try:
        cwd = os.getcwd()
        # 检查是否 git 仓库
        result = run_command(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd, timeout=5,
        )
        if not result.ok:
            return None

        # 获取分支
        branch_result = run_command(
            ["git", "branch", "--show-current"],
            cwd=cwd, timeout=5,
        )
        branch = branch_result.stdout.strip()

        # 获取状态
        status_result = run_command(
            ["git", "status", "--short"],
            cwd=cwd, timeout=5,
        )
        status = status_result.stdout.strip()

        # 获取最近提交
        log_result = run_command(
            ["git", "log", "--oneline", "-n", "5"],
            cwd=cwd, timeout=5,
        )
        log = log_result.stdout.strip()

        # 截断过长的状态
        max_chars = 1000
        if len(status) > max_chars:
            status = status[:max_chars] + "\n... (truncated)"

        parts = [
            f"This is the git status at the start of the conversation.",
            f"Current branch: {branch}",
            f"Status:\n{status or '(clean)'}",
            f"Recent commits:\n{log}",
        ]
        return "\n\n".join(parts)
    except Exception:
        return None


# ======================================================================
# 系统提示词组装
# ======================================================================

def fetch_system_prompt_parts(
    tools: List[Any],
    main_loop_model: str = "",
    custom_system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """组装系统提示词的三个部分。

    对应 CCB 的 fetchSystemPromptParts()。

    Returns:
        (default_system_prompt, user_context, system_context)
    """
    # 默认系统提示词
    if custom_system_prompt:
        default_prompt = [custom_system_prompt]
    else:
        default_prompt = [DEFAULT_SYSTEM_PROMPT]

    # 用户上下文
    user_context = get_user_context()

    # 系统上下文
    system_context = get_system_context()

    # 追加自定义提示词
    if append_system_prompt:
        default_prompt.append(append_system_prompt)

    return default_prompt, user_context, system_context


def as_system_prompt(parts: List[str]) -> str:
    """将多个提示词部分合并为单个系统提示词字符串。

    对应 CCB 的 asSystemPrompt()。
    """
    return "\n\n".join(p for p in parts if p)


def append_system_context(
    system_prompt: str,
    system_context: Dict[str, str],
) -> str:
    """将系统上下文追加到系统提示词。

    对应 CCB 的 appendSystemContext()。
    """
    if not system_context:
        return system_prompt
    context_parts = []
    for key, value in system_context.items():
        context_parts.append(f"## {key}\n{value}")
    return system_prompt + "\n\n" + "\n\n".join(context_parts)


def prepend_user_context(
    messages: List[Dict[str, Any]],
    user_context: Dict[str, str],
) -> List[Dict[str, Any]]:
    """将用户上下文前置于消息列表。

    对应 CCB 的 prependUserContext()。
    """
    if not user_context:
        return messages
    context_parts = []
    for key, value in user_context.items():
        context_parts.append(f"[{key}]\n{value}")
    context_text = "\n\n".join(context_parts)

    result = []
    for msg in messages:
        if msg.get("role") == "system":
            # 合并到 system 消息
            result.append({
                "role": "system",
                "content": msg["content"] + "\n\n" + context_text,
            })
        else:
            result.append(msg)
    if not any(m.get("role") == "system" for m in result):
        result.insert(0, {"role": "system", "content": context_text})
    return result


def clear_context_caches():
    """清除上下文缓存（当环境变化时调用）。"""
    get_user_context.cache_clear()
    get_system_context.cache_clear()
