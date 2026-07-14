from __future__ import annotations

from pathlib import Path

from aerospace_agent.langgraph_agent.agent_core.context_assembler import (
    MemoryContextAssembler,
)
from aerospace_agent.langgraph_agent.agent_core.models import CheckpointRef
from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService
from aerospace_agent.langgraph_agent.agent_core.session_memory import SessionMemoryService


def _seed(tmp_path: Path, thread_id: str, content: str, checkpoint_id: str = "cp"):
    project = ProjectIdentityService(tmp_path)
    status = project.status()
    project_id = status.project_id or project.initialize().project_id
    assert project_id is not None
    valid = {checkpoint_id}
    session = SessionMemoryService(
        tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        thread_id=thread_id,
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid,
    )
    session.remember(
        kind="constraint",
        content=content,
        source_checkpoints=[
            CheckpointRef(
                project_id=project_id, thread_id=thread_id, checkpoint_id=checkpoint_id
            )
        ],
        source_content_hash="a" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )
    return project, project_id, valid


def test_context_assembler_returns_project_then_summary_then_same_thread_memory(tmp_path) -> None:
    project, project_id, valid = _seed(
        tmp_path, "thread-a", "Orbit output must use GCRF"
    )
    (tmp_path / "memory/project/PROJECT.md").write_text(
        "Approved project orbit frame is GCRF", encoding="utf-8"
    )
    project.reindex()
    session = SessionMemoryService(
        tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        thread_id="thread-a",
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid,
    )
    session.save_summary(
        current_goal="Review orbit result",
        confirmed_constraints=["Use SI"],
        decisions=[],
        completed_items=[],
        open_items=["Validate frame"],
        artifacts=[],
        assumptions=[],
        source_checkpoints=[
            CheckpointRef(project_id=project_id, thread_id="thread-a", checkpoint_id="cp")
        ],
    )
    assembler = MemoryContextAssembler(
        project_memory=project,
        session_database_path=tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid,
    )

    sections = assembler.assemble(thread_id="thread-a", query="orbit GCRF")

    assert sections.project_memory and "PROJECT.md" in sections.project_memory[0]
    assert sections.session_summary is not None
    assert sections.session_summary.current_goal == "Review orbit result"
    assert [item.content for item in sections.session_memories] == [
        "Orbit output must use GCRF"
    ]
    assert sections.prompt_sections()[0].startswith("[PROJECT MEMORY]")
    assert sections.prompt_sections()[-1].startswith("[SESSION MEMORY]")


def test_context_assembler_never_reads_other_thread_or_untraceable_memory(tmp_path) -> None:
    project, project_id, _ = _seed(tmp_path, "thread-a", "secret thread A", "cp-a")
    _, _, valid_b = _seed(tmp_path, "thread-b", "visible thread B", "cp-b")
    assembler = MemoryContextAssembler(
        project_memory=project,
        session_database_path=tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid_b,
    )

    sections = assembler.assemble(thread_id="thread-b", query="thread")

    assert [item.content for item in sections.session_memories] == ["visible thread B"]
    assert all("secret thread A" not in text for text in sections.prompt_sections())


def test_context_assembler_injects_stable_preferences_for_unrelated_followups(tmp_path):
    project, project_id, valid = _seed(tmp_path, "thread-a", "Orbit output must use GCRF")
    session = SessionMemoryService(
        tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        thread_id="thread-a",
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid,
    )
    session.remember(
        kind="preference",
        content="以后都称呼我爸爸",
        source_checkpoints=[
            CheckpointRef(project_id=project_id, thread_id="thread-a", checkpoint_id="cp")
        ],
        source_content_hash="b" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )
    assembler = MemoryContextAssembler(
        project_memory=project,
        session_database_path=tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid,
    )

    sections = assembler.assemble(thread_id="thread-a", query="你好")

    assert any(item.content == "以后都称呼我爸爸" for item in sections.session_memories)


def test_empty_query_does_not_dump_all_session_memory(tmp_path) -> None:
    project, project_id, valid = _seed(tmp_path, "thread", "sensitive fact")
    assembler = MemoryContextAssembler(
        project_memory=project,
        session_database_path=tmp_path / "data/langgraph/session_memory.sqlite",
        project_id=project_id,
        checkpoint_validator=lambda checkpoint, _digest: checkpoint.checkpoint_id in valid,
    )
    assert assembler.assemble(thread_id="thread", query="").session_memories == []
