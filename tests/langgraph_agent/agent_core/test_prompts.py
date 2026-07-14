from __future__ import annotations

from aerospace_agent.langgraph_agent.prompts import sanitize_assistant_answer


def test_model_vendor_self_identification_is_replaced_by_aerospace_identity() -> None:
    answer = "您好！我是 Qwythos，由 Empero AI 打造的航天领域智能助手。\n\n- 航天知识问答"

    sanitized = sanitize_assistant_answer(answer)

    assert sanitized.startswith("我是您航天领域共同学习进步的AI助手")
    assert "航天知识问答" in sanitized
    assert "Qwythos" not in sanitized
    assert "Empero AI" not in sanitized
