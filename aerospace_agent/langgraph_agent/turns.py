"""Explicit per-turn lifecycle for the LangGraph Agent Core.

The graph remains responsible for domain routing and execution.  This module
only makes the outer turn contract observable and keeps command shortcuts out
of the model/RAG path.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TurnState(StrEnum):
    RESTORE = "restore"
    COMPACT = "compact"
    COMMAND = "command"
    BUILD = "build"
    RUN = "run"
    SAVE = "save"
    RESPOND = "respond"
    DONE = "done"


class CommandDecision(BaseModel):
    """Deterministic slash-command result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matched: bool = False
    command: str | None = None
    arguments: tuple[str, ...] = ()
    response: str | None = None


class TurnContext(BaseModel):
    """Checkpoint-safe metadata for one isolated conversation turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    state: TurnState = TurnState.RESTORE
    state_history: list[TurnState] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    restored_checkpoint_id: str | None = None
    response: str = ""
    command: str | None = None
    shortcut: bool = False

    def advance(self, state: TurnState) -> "TurnContext":
        history = list(self.state_history)
        if not history or history[-1] != state:
            history.append(state)
        return self.model_copy(update={"state": state, "state_history": history})

    def with_context(
        self,
        *,
        thread_id: str | None = None,
        **values: Any,
    ) -> "TurnContext":
        if thread_id is not None and str(thread_id) != self.thread_id:
            raise ValueError("thread_id cannot change within a turn")
        merged = dict(self.context)
        merged.update(values)
        return self.model_copy(update={"context": merged})

    def with_response(self, response: Any) -> "TurnContext":
        return self.model_copy(update={"response": str(response or "")})


class CommandRouter:
    """Parse only safe, explicit slash commands before model execution."""

    _pattern = re.compile(r"^\s*/(?P<command>stop|model|skip)(?:\s+(?P<args>.*))?\s*$", re.I)

    def dispatch(self, message: str) -> CommandDecision:
        match = self._pattern.match(str(message))
        if match is None:
            return CommandDecision()
        command = match.group("command").lower()
        raw_args = (match.group("args") or "").strip()
        arguments = tuple(raw_args.split()) if raw_args else ()
        if command == "stop":
            response = "已停止当前回合。"
        elif command == "model":
            response = (
                f"已请求切换模型：{arguments[0]}。"
                if arguments
                else "当前未提供模型名称。"
            )
        else:
            response = "已跳过当前回合的后续执行。"
        return CommandDecision(
            matched=True,
            command=command,
            arguments=arguments,
            response=response,
        )


Callback = Callable[[TurnContext], TurnContext]
CommandHandler = Callable[[TurnContext, CommandDecision], str | None]


class AgentLoop:
    """Run the explicit RESTORE→...→DONE turn lifecycle."""

    def __init__(self, *, command_router: CommandRouter | None = None) -> None:
        self.command_router = command_router or CommandRouter()

    @staticmethod
    def _stage(context: TurnContext, state: TurnState, callback: Callback) -> TurnContext:
        return callback(context.advance(state))

    def run(
        self,
        context: TurnContext,
        *,
        restore: Callback,
        compact: Callback,
        build: Callback,
        execute: Callback,
        save: Callback,
        respond: Callback,
        command_handler: CommandHandler | None = None,
    ) -> TurnContext:
        current = self._stage(context, TurnState.RESTORE, restore)
        current = self._stage(current, TurnState.COMPACT, compact)
        current = current.advance(TurnState.COMMAND)
        decision = self.command_router.dispatch(current.user_message)
        if decision.matched:
            handled_response = (
                command_handler(current, decision) if command_handler is not None else None
            )
            current = current.model_copy(
                update={
                    "command": decision.command,
                    "shortcut": True,
                    "response": handled_response or decision.response or "",
                }
            )
            return current.advance(TurnState.DONE)
        current = self._stage(current, TurnState.BUILD, build)
        current = self._stage(current, TurnState.RUN, execute)
        current = self._stage(current, TurnState.SAVE, save)
        current = self._stage(current, TurnState.RESPOND, respond)
        return current.advance(TurnState.DONE)


__all__ = ["AgentLoop", "CommandDecision", "CommandRouter", "TurnContext", "TurnState"]
