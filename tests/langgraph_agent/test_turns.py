from __future__ import annotations

from aerospace_agent.langgraph_agent.turns import (
    AgentLoop,
    CommandRouter,
    TurnContext,
    TurnState,
)


def test_agent_loop_runs_explicit_lifecycle_in_order() -> None:
    events: list[str] = []
    loop = AgentLoop(command_router=CommandRouter())
    context = TurnContext(
        project_id="project-a",
        thread_id="thread-a",
        run_id="run-a",
        user_message="hello",
    )

    result = loop.run(
        context,
        restore=lambda item: events.append("restore") or item,
        compact=lambda item: events.append("compact") or item,
        build=lambda item: events.append("build") or item,
        execute=lambda item: events.append("run") or item.with_response("done"),
        save=lambda item: events.append("save") or item,
        respond=lambda item: events.append("respond") or item,
    )

    assert events == ["restore", "compact", "build", "run", "save", "respond"]
    assert result.state == TurnState.DONE
    assert result.state_history == [
        TurnState.RESTORE,
        TurnState.COMPACT,
        TurnState.COMMAND,
        TurnState.BUILD,
        TurnState.RUN,
        TurnState.SAVE,
        TurnState.RESPOND,
        TurnState.DONE,
    ]
    assert result.response == "done"


def test_command_router_shortcuts_stop_without_running_model() -> None:
    events: list[str] = []
    loop = AgentLoop(command_router=CommandRouter())
    context = TurnContext(
        project_id="project-a",
        thread_id="thread-a",
        run_id="run-a",
        user_message="/stop",
    )

    result = loop.run(
        context,
        restore=lambda item: events.append("restore") or item,
        compact=lambda item: events.append("compact") or item,
        build=lambda item: events.append("build") or item,
        execute=lambda item: events.append("run") or item.with_response("bad"),
        save=lambda item: events.append("save") or item,
        respond=lambda item: events.append("respond") or item,
    )

    assert events == ["restore", "compact"]
    assert result.shortcut is True
    assert result.command == "stop"
    assert result.state == TurnState.DONE
    assert result.response == "已停止当前回合。"


def test_turn_context_rejects_cross_thread_context() -> None:
    context = TurnContext(
        project_id="project-a",
        thread_id="thread-a",
        run_id="run-a",
        user_message="hello",
    )

    try:
        context.with_context(thread_id="thread-b")
    except ValueError as exc:
        assert "thread_id" in str(exc)
    else:
        raise AssertionError("cross-thread context must be rejected")
