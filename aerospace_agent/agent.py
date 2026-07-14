"""Lifecycle-safe facade around the deterministic aerospace LangGraph."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from .checkpointer import (
    DEFAULT_CHECKPOINT_DB,
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    get_checkpointer,
    list_saved_threads,
)
from .evolution import create_evolution_engine
from .langgraph_agent.services.evolution import EvolutionService
from .langgraph_agent.services.evolution_policy import EvolutionPolicy
from .graph import ServiceBundle, build_aerospace_graph, build_simple_graph
from .schema import AgentInput, AgentOutput, IntentType, RunStatus
from .state import create_initial_state


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {}


class LangGraphAerospaceAgent:
    """Own a graph, its services, and its checkpointer for one agent lifetime."""

    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        model_name: str = "qwen3-vl",
        rag: Any = None,
        available_tools: Optional[Dict[str, Any]] = None,
        checkpoint_backend: str = "sqlite",
        checkpoint_db_path: Optional[str | Path] = None,
        evolution_db_path: Optional[str | Path] = None,
        max_steps: int = 15,
        max_recursion_depth: int = 40,
        cycle_max_repeats: int = 3,
        use_llm_intent: bool = False,
        mode: str = "full",
        *,
        settings: Any = None,
        services: ServiceBundle | None = None,
        interrupt_before: Iterable[str] | None = None,
        checkpointer: Any = None,
    ):
        self.settings = settings
        runtime = getattr(settings, "runtime", None)
        checkpoint_settings = getattr(settings, "checkpoint", None)
        llm_settings = getattr(settings, "llm", None)
        if settings is not None:
            if llm_endpoint is None:
                llm_endpoint = getattr(llm_settings, "endpoint", None)
            if model_name == "qwen3-vl":
                model_name = str(getattr(llm_settings, "model", model_name))
            max_steps = int(getattr(runtime, "max_steps", max_steps))
            max_recursion_depth = int(getattr(runtime, "recursion_limit", max_recursion_depth))
            cycle_max_repeats = int(getattr(runtime, "cycle_max_repeats", cycle_max_repeats))
            if checkpoint_db_path is None:
                checkpoint_db_path = getattr(checkpoint_settings, "path", None)
            if checkpoint_backend == "sqlite":
                checkpoint_backend = str(getattr(checkpoint_settings, "backend", checkpoint_backend))

        self.model_name = model_name
        self.llm_endpoint = llm_endpoint
        self.rag = rag
        self.available_tools = available_tools or {}
        self.max_steps = int(max_steps)
        self.recursion_limit = int(max_recursion_depth)
        self.max_recursion_depth = self.recursion_limit
        self.cycle_max_repeats = int(cycle_max_repeats)
        self.use_llm_intent = bool(use_llm_intent)
        self.mode = mode
        self.interrupt_before = tuple(interrupt_before or ())
        self.checkpoint_backend = checkpoint_backend
        self._checkpoint_db_path = str(checkpoint_db_path or DEFAULT_CHECKPOINT_DB)
        self._checkpoint_ctx = None
        self._closed = False
        self._config: Dict[str, Any] = {"configurable": {"thread_id": "default"}}

        self.llm = self._create_llm(llm_endpoint, model_name) if llm_endpoint else None
        self.services = services or ServiceBundle(
            knowledge=rag,
            mcp_gateway=available_tools,
            llm=self.llm,
            model_name=model_name,
            endpoint=llm_endpoint or "",
        )
        self.checkpointer = checkpointer
        self._init_checkpointer()
        self.evolution = create_evolution_engine(db_path=evolution_db_path)
        evolution_settings = getattr(settings, "evolution", None)
        evolution_workspace = getattr(getattr(settings, "knowledge", None), "workspace", None) or Path.cwd()
        evolution_data_dir = getattr(evolution_settings, "data_dir", None) if evolution_settings is not None else None
        evolution_roots = getattr(evolution_settings, "allowed_roots", None) if evolution_settings is not None else None
        self.evolution_service = EvolutionService(
            workspace=evolution_workspace,
            data_dir=evolution_data_dir,
            allowed_roots=evolution_roots,
            policy=None if evolution_settings is None else EvolutionPolicy(
                enabled=bool(getattr(evolution_settings, "enabled", True)),
                idle_minutes=int(getattr(evolution_settings, "idle_minutes", 10)),
                min_turns=int(getattr(evolution_settings, "min_turns", 6)),
                allowed_roots=evolution_roots,
            ),
        )
        self.graph = self._build_graph()

    def _create_llm(self, endpoint: str | None, model_name: str):
        return SimpleLLMClient(endpoint=endpoint, model=model_name) if endpoint else None

    def _init_checkpointer(self) -> None:
        if self.checkpointer is not None:
            return
        if str(self.checkpoint_backend).lower() == "sqlite":
            self._checkpoint_ctx = create_sqlite_checkpointer(self._checkpoint_db_path)
            self.checkpointer = self._checkpoint_ctx.__enter__()
        else:
            self.checkpointer = create_memory_checkpointer()

    def _close_checkpointer(self) -> None:
        ctx, self._checkpoint_ctx = self._checkpoint_ctx, None
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass

    def _build_graph(self):
        kwargs = dict(
            checkpointer=self.checkpointer,
            max_steps=self.max_steps,
            max_repeats=self.cycle_max_repeats,
            max_recursion_depth=self.recursion_limit,
            use_llm_intent=self.use_llm_intent,
            services=self.services,
        )
        if self.mode == "simple":
            return build_simple_graph(checkpointer=self.checkpointer, services=self.services)
        return build_aerospace_graph(**kwargs)

    def _config_for(self, thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": str(thread_id)}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        # ``recursion_limit`` is an invocation option, never a state channel.
        return {"configurable": configurable, "recursion_limit": self.recursion_limit}

    @staticmethod
    def _checkpoint_id(snapshot: Any) -> str | None:
        config = getattr(snapshot, "config", None) or {}
        return (_as_dict(config.get("configurable", {}))).get("checkpoint_id")

    def _snapshot(self, thread_id: str, checkpoint_id: str | None = None):
        return self.graph.get_state(self._config_for(thread_id, checkpoint_id))

    def _input_for_run(self, user_message: str, thread_id: str, context: dict[str, Any] | None) -> dict[str, Any]:
        try:
            snapshot = self._snapshot(thread_id)
            existing = bool(snapshot and getattr(snapshot, "values", None))
        except Exception:
            existing = False
        if existing:
            # ``add_messages`` merges this delta with the durable history.
            messages: list[Any] = [HumanMessage(content=user_message)]
            if context:
                messages.append(AIMessage(content=f"[context] {json.dumps(context, ensure_ascii=False)}"))
            return {"messages": messages, "run_id": uuid.uuid4().hex}
        state = create_initial_state(thread_id=thread_id, run_id=uuid.uuid4().hex, max_steps=self.max_steps, max_cycles=self.max_steps)
        state["messages"] = [HumanMessage(content=user_message)]
        if context:
            state["messages"].append(AIMessage(content=f"[context] {json.dumps(context, ensure_ascii=False)}"))
        return state

    def _output_from_snapshot(
        self,
        snapshot: Any,
        *,
        thread_id: str,
        started: float,
        forced_status: str | None = None,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> AgentOutput:
        values = _as_dict(getattr(snapshot, "values", {}) if snapshot is not None else {})
        checkpoint_id = self._checkpoint_id(snapshot) if snapshot is not None else None
        status = forced_status or str(values.get("status") or ("interrupted" if getattr(snapshot, "next", ()) else "partial"))
        if error_category:
            status = "limit_reached" if error_category == "graph_recursion_limit" else "error"
        elapsed = max(0.0, (time.perf_counter() - started) * 1000.0)
        metrics = dict(values.get("metrics", {}) or {})
        category = error_category or metrics.get("error_category")
        metrics.update({
            "run_id": str(values.get("run_id", "")),
            "thread_id": str(thread_id),
            "checkpoint_id": checkpoint_id,
            "model_name": str(getattr(self.services, "model_name", "") or self.model_name),
            "endpoint": str(getattr(self.services, "endpoint", "") or self.llm_endpoint or ""),
            "duration_ms": elapsed,
            "total_duration_ms": float(metrics.get("total_duration_ms", 0.0) or 0.0),
            "rag_hits": int(metrics.get("rag_hits", len(values.get("evidence", []) or [])) or 0),
            "cycles": int(values.get("cycle_count", 0) or 0),
            "warnings": list(values.get("warnings", []) or []),
            "errors": list(values.get("errors", []) or []),
        })
        if category:
            metrics["error_category"] = category
        raw_errors = list(values.get("errors", []) or [])
        if error_message:
            raw_errors.append({"category": error_category or "agent_error", "message": error_message})
        intent_raw = str(values.get("intent", "general"))
        try:
            intent = IntentType(intent_raw)
        except Exception:
            intent = IntentType.GENERAL
        answer = str(values.get("final_answer", "") or "")
        if status == "interrupted":
            answer = ""
        return AgentOutput(
            status=status,
            answer=answer,
            intent=intent,
            intent_confidence=float(values.get("intent_confidence", 0.0) or 0.0),
            citations=list(values.get("citations", values.get("evidence", [])) or []),
            tool_results=list(values.get("tool_results", []) or []),
            steps=int(values.get("step_count", values.get("recursion_depth", 0)) or 0),
            cycle_triggers=int(values.get("cycle_count", 0) or 0),
            checkpoint_id=checkpoint_id,
            warnings=[str(item) for item in values.get("warnings", []) or []],
            errors=raw_errors,
            metrics=metrics,
        )

    def run(self, user_message: str, thread_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        thread_id = str(thread_id or uuid.uuid4().hex[:8])
        self._config = self._config_for(thread_id)
        started = time.perf_counter()
        try:
            AgentInput(user_message=user_message, thread_id=thread_id, max_steps=self.max_steps, recursion_limit=max(self.recursion_limit, self.max_steps + 1), context=context or {})
            payload = self._input_for_run(user_message, thread_id, context)
            self.graph.invoke(payload, config=self._config, interrupt_before=self.interrupt_before)
            snapshot = self._snapshot(thread_id)
            if getattr(snapshot, "next", ()):
                return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started, forced_status="interrupted")
            return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started)
        except GraphRecursionError as exc:
            try:
                snapshot = self._snapshot(thread_id)
            except Exception:
                snapshot = None
            return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started, error_category="graph_recursion_limit", error_message=str(exc))
        except OSError as exc:
            # A failed saver write must never be reported as a successful run.
            return self._output_from_snapshot(None, thread_id=thread_id, started=started, error_category="checkpoint_write_error", error_message=str(exc))
        except Exception as exc:
            try:
                snapshot = self._snapshot(thread_id)
            except Exception:
                snapshot = None
            return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started, error_category="agent_error", error_message=str(exc))

    def resume_execution(self, thread_id: str) -> AgentOutput:
        """Resume the saved graph at ``snapshot.next`` without a new input."""
        started = time.perf_counter()
        try:
            snapshot = self._snapshot(thread_id)
            if snapshot is None or not getattr(snapshot, "next", ()):
                return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started, error_category="resume_unavailable", error_message="no pending graph execution")
            self.graph.invoke(None, config=self._config_for(thread_id), interrupt_before=[])
            return self._output_from_snapshot(self._snapshot(thread_id), thread_id=thread_id, started=started)
        except GraphRecursionError as exc:
            return self._output_from_snapshot(self._snapshot(thread_id), thread_id=thread_id, started=started, error_category="graph_recursion_limit", error_message=str(exc))
        except OSError as exc:
            return self._output_from_snapshot(None, thread_id=thread_id, started=started, error_category="checkpoint_write_error", error_message=str(exc))
        except Exception as exc:
            return self._output_from_snapshot(self._snapshot(thread_id), thread_id=thread_id, started=started, error_category="agent_error", error_message=str(exc))

    def resume(self, thread_id: str) -> bool:
        try:
            return bool(self._snapshot(thread_id))
        except Exception:
            return False

    def stream(self, user_message: str, thread_id: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        thread_id = str(thread_id or uuid.uuid4().hex[:8])
        config = self._config_for(thread_id)
        payload = self._input_for_run(user_message, thread_id, None)
        yield from self.graph.stream(payload, config=config, stream_mode="values", interrupt_before=self.interrupt_before)

    def stream_text(self, user_message: str, thread_id: Optional[str] = None) -> Generator[str, None, None]:
        if self.llm is None:
            yield self.run(user_message, thread_id=thread_id).answer
            return
        yield from self.llm.stream_chat(user_message)

    def list_conversations(self) -> List[str]:
        if str(self.checkpoint_backend).lower() == "sqlite":
            return list_saved_threads(self._checkpoint_db_path)
        found: set[str] = set()
        try:
            for item in self.checkpointer.list(None):
                cfg = _as_dict(getattr(item, "config", {})).get("configurable", {})
                if cfg.get("thread_id"):
                    found.add(str(cfg["thread_id"]))
        except Exception:
            pass
        return sorted(found)

    def get_conversation_state(self, thread_id: str, checkpoint_id: str | None = None):
        return self._snapshot(thread_id, checkpoint_id)

    def get_state(self, thread_id: str, checkpoint_id: str | None = None):
        return self.get_conversation_state(thread_id, checkpoint_id)

    def get_checkpoint_history(self, thread_id: str) -> list[dict[str, Any]]:
        snapshots = list(self.graph.get_state_history(self._config_for(thread_id)))
        result: list[dict[str, Any]] = []
        for index, snapshot in enumerate(snapshots):
            cp_id = self._checkpoint_id(snapshot)
            metadata = _as_dict(getattr(snapshot, "metadata", {}))
            created_at = getattr(snapshot, "created_at", None) or metadata.get("created_at") or cp_id or index
            result.append({
                "checkpoint_id": cp_id,
                "created_at": created_at,
                "parent_checkpoint_id": self._checkpoint_id(getattr(snapshot, "parent_config", None)),
                "values": getattr(snapshot, "values", {}),
                "next": tuple(getattr(snapshot, "next", ()) or ()),
                "metadata": metadata,
                "snapshot": snapshot,
            })
        return result

    def get_state_history(self, thread_id: str):
        return self.get_checkpoint_history(thread_id)

    def replay_checkpoint(self, thread_id: str, checkpoint_id: str | None = None):
        return self.get_conversation_state(thread_id, checkpoint_id)

    replay = replay_checkpoint

    def fork_from_checkpoint(self, checkpoint_id: str, *, new_thread_id: str, source_thread_id: str | None = None):
        source = source_thread_id
        if source is None:
            for candidate in self.list_conversations():
                if any(item.get("checkpoint_id") == checkpoint_id for item in self.get_checkpoint_history(candidate)):
                    source = candidate
                    break
        if source is None:
            raise ValueError(f"checkpoint not found: {checkpoint_id}")
        snapshot = self.get_conversation_state(source, checkpoint_id)
        values = dict(getattr(snapshot, "values", {}) or {})
        values["thread_id"] = new_thread_id
        values["run_id"] = uuid.uuid4().hex
        values["status"] = ""
        values["final_answer"] = ""
        values["is_complete"] = False
        self.graph.invoke(values, config=self._config_for(new_thread_id), interrupt_before=[])
        return self.get_conversation_state(new_thread_id)

    def get_evolution_summary(self) -> Dict[str, Any]:
        return self.evolution.get_evolution_summary()

    def evolve(self, proposal: Any, **kwargs: Any):
        return self.evolution_service.apply(proposal, **kwargs)

    def evolve_due(self, proposal: Any = None, **kwargs: Any):
        return self.evolution_service.evolve_due(proposal, **kwargs)

    def get_available_tools(self) -> List[str]:
        return list(self.available_tools.keys())

    def reset(self) -> None:
        self._config = self._config_for(uuid.uuid4().hex[:8])

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close_checkpointer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<LangGraphAerospaceAgent model={self.model_name} mode={self.mode} checkpointer={type(self.checkpointer).__name__}>"


class SimpleLLMClient:
    """Tiny OpenAI-compatible HTTP client retained for existing callers."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8000/v1", model: str = "qwen3-vl", api_key: str = "not-needed", timeout: float = 60.0):
        self.endpoint, self.model, self.api_key, self.timeout = endpoint.rstrip("/"), model, api_key, timeout

    def chat(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> str:
        import urllib.request
        payload = {"model": self.model, "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "stream": False}
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode())["choices"][0]["message"]["content"]

    def stream_chat(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.7):
        import urllib.request
        payload = {"model": self.model, "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for line in response:
                text = line.decode().strip()
                if text.startswith("data: ") and text != "data: [DONE]":
                    try:
                        content = json.loads(text[6:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except Exception:
                        content = ""
                    if content:
                        yield content

    def is_available(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.endpoint}/models", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


def create_agent(llm_endpoint: str = "http://127.0.0.1:8000/v1", model_name: str = "qwen3-vl", checkpoint_backend: str = "sqlite", mode: str = "full", **kwargs) -> LangGraphAerospaceAgent:
    return LangGraphAerospaceAgent(llm_endpoint=llm_endpoint, model_name=model_name, checkpoint_backend=checkpoint_backend, mode=mode, **kwargs)


__all__ = ["LangGraphAerospaceAgent", "SimpleLLMClient", "create_agent"]
