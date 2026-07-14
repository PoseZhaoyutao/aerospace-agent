"""Shared model instructions for the aerospace assistant surface."""

from __future__ import annotations


AEROSPACE_ASSISTANT_IDENTITY = (
    "我是您航天领域共同学习进步的AI助手，可以帮助您完成普通对话、航天知识问答、"
    "工作规划、工具调用以及跨领域航天任务编排。"
    "我会区分已确认事实、用户约束和假设；只有在低置信度、用户要求依据或规划器主动请求核实时才使用私域知识库。"
    "涉及文件、终端或其他会改变状态的操作，我会先请求确认；只读操作可以直接执行。"
    "请不要把我描述为其他产品、模型或公司身份，也不要声称尚未执行的操作已经完成。"
    "回答应直接服务于用户当前请求，保持航天工程语境和可核查性。"
)

AEROSPACE_ASSISTANT_INTRO = (
    "我是您航天领域共同学习进步的AI助手，可以帮助您完成普通对话、航天知识问答、"
    "工作规划、工具调用以及跨领域航天任务编排。"
)


def sanitize_assistant_answer(answer: str) -> str:
    """Remove model-vendor self-identification from user-visible output.

    Local models may ignore an otherwise correct system prompt.  Drop only
    lines containing the forbidden vendor names, preserve the remaining useful
    response, and provide the canonical aerospace introduction if nothing is
    left.
    """

    text = str(answer or "").strip()
    if not text:
        return text
    forbidden = ("qwythos", "empero ai")
    if not any(token in text.casefold() for token in forbidden):
        return text
    kept_lines = [
        line for line in text.splitlines()
        if not any(token in line.casefold() for token in forbidden)
    ]
    body = "\n".join(kept_lines).strip()
    return f"{AEROSPACE_ASSISTANT_INTRO}\n\n{body}" if body else AEROSPACE_ASSISTANT_INTRO


__all__ = [
    "AEROSPACE_ASSISTANT_IDENTITY",
    "AEROSPACE_ASSISTANT_INTRO",
    "sanitize_assistant_answer",
]
