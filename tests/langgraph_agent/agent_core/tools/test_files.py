from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent_core.journal import OperationJournal
from aerospace_agent.langgraph_agent.agent_core.tools.files import FileService


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def journal(tmp_path: Path) -> OperationJournal:
    return OperationJournal(tmp_path / "state" / "operations.sqlite3")


@pytest.fixture
def service(workspace: Path, journal: OperationJournal) -> FileService:
    return FileService(workspace, journal=journal)


def test_read_lines_list_info_and_search_are_bounded_and_read_only(
    service: FileService, workspace: Path
) -> None:
    notes = workspace / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\nneedle here\nomega\n", encoding="utf-8")
    (notes / "b.txt").write_text("another needle\n", encoding="utf-8")

    read_result = service.read("notes/a.txt", max_bytes=10)
    lines_result = service.read_lines("notes/a.txt", start_line=2, end_line=2)
    list_result = service.list("notes", max_results=1)
    info_result = service.info("notes/a.txt")
    stat_result = service.stat("notes/a.txt")
    search_result = service.search("notes", "needle", max_results=1)

    assert read_result.status == "success"
    assert read_result.recovery_class == "read_only"
    assert read_result.result["truncated"] is True
    assert read_result.result["sha256"] == hashlib.sha256(
        (notes / "a.txt").read_bytes()
    ).hexdigest()
    assert lines_result.result["content"].splitlines() == ["needle here"]
    assert list_result.result["truncated"] is True
    assert len(list_result.result["entries"]) == 1
    assert info_result.result["type"] == "file"
    assert stat_result.result == info_result.result
    assert search_result.result["truncated"] is True
    assert len(search_result.result["matches"]) == 1


def test_atomic_mutations_record_preimages_and_are_reversible(
    service: FileService, workspace: Path, journal: OperationJournal
) -> None:
    mkdir_result = service.mkdir("work", operation_id="mkdir-op")
    write_result = service.write("work/a.txt", "old", operation_id="write-op")
    overwrite_result = service.write(
        "work/a.txt", "new", overwrite=True, operation_id="overwrite-op"
    )
    append_result = service.append("work/a.txt", "+tail", operation_id="append-op")
    copy_result = service.copy("work/a.txt", "work/b.txt", operation_id="copy-op")
    move_result = service.move("work/b.txt", "work/c.txt", operation_id="move-op")
    blocked_delete = service.delete("work/c.txt", operation_id="delete-blocked")
    delete_result = service.delete(
        "work/c.txt", confirmed=True, operation_id="delete-op"
    )

    assert {result.recovery_class for result in (
        mkdir_result,
        write_result,
        overwrite_result,
        append_result,
        copy_result,
        move_result,
        delete_result,
    )} == {"reversible"}
    assert blocked_delete.status == "blocked"
    assert blocked_delete.error is not None
    assert blocked_delete.error.code == "confirmation_required"
    assert (workspace / "work/a.txt").read_text(encoding="utf-8") == "new+tail"
    assert not (workspace / "work/c.txt").exists()

    overwrite_preimage = journal.list_preimages("overwrite-op")[0]
    assert overwrite_preimage["kind"] == "file"
    assert overwrite_preimage["sha256"] == hashlib.sha256(b"old").hexdigest()
    assert Path(overwrite_preimage["backup_path"]).read_bytes() == b"old"

    append_record = journal.get("append-op")
    assert append_record is not None
    assert append_record["metadata"]["original_length"] == 3
    assert append_record["metadata"]["original_sha256"] == hashlib.sha256(
        b"new"
    ).hexdigest()
    assert all(
        journal.get(operation_id)["status"] == "completed"
        for operation_id in (
            "mkdir-op",
            "write-op",
            "overwrite-op",
            "append-op",
            "copy-op",
            "move-op",
            "delete-op",
        )
    )


def test_reversible_file_operation_rolls_back_only_when_postimage_still_matches(
    service: FileService, workspace: Path, journal: OperationJournal
) -> None:
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    result = service.write(
        "value.txt", "after", overwrite=True, operation_id="rollback-op"
    )
    assert result.status == "success"

    journal.rollback("rollback-op")

    assert target.read_text(encoding="utf-8") == "before"
    assert journal.get("rollback-op")["status"] == "rolled_back"

    service.write("value.txt", "next", overwrite=True, operation_id="drift-op")
    target.write_text("concurrent-change", encoding="utf-8")
    with pytest.raises(RuntimeError, match="postimage"):
        journal.rollback("drift-op")
    assert target.read_text(encoding="utf-8") == "concurrent-change"


def test_existing_important_file_overwrite_and_move_require_confirmation(
    service: FileService, workspace: Path
) -> None:
    agents = workspace / "AGENTS.md"
    agents.write_text("original", encoding="utf-8")

    blocked_write = service.write("AGENTS.md", "changed", overwrite=True)
    blocked_move = service.move("AGENTS.md", "archive/AGENTS.md")

    assert blocked_write.status == "blocked"
    assert blocked_write.error is not None
    assert blocked_write.error.code == "confirmation_required"
    assert blocked_move.status == "blocked"
    assert agents.read_text(encoding="utf-8") == "original"

    allowed = service.write(
        "AGENTS.md", "changed", overwrite=True, confirmed=True
    )
    assert allowed.status == "success"
    assert allowed.recovery_class == "reversible"


def test_existing_target_conflict_is_not_reclassified_as_invalid_arguments(
    service: FileService, workspace: Path
) -> None:
    (workspace / "value.txt").write_text("existing", encoding="utf-8")

    result = service.write("value.txt", "replacement", overwrite=False)

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "conflict"
    assert (workspace / "value.txt").read_text(encoding="utf-8") == "existing"


def test_symlink_escape_is_rejected_for_reads_and_writes(
    service: FileService, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = workspace / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"symlink creation is unavailable: {exc}")
        junction = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        if junction.returncode != 0:
            pytest.skip(f"symlink and junction creation are unavailable: {exc}")

    try:
        read_result = service.read("escape/secret.txt")
        write_result = service.write("escape/new.txt", "no")

        assert read_result.status == "blocked"
        assert read_result.error is not None
        assert read_result.error.code == "path_outside_workspace"
        assert write_result.status == "blocked"
        assert not (outside / "new.txt").exists()
    finally:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            link.rmdir()


def test_failed_journal_preparation_does_not_change_existing_file(
    workspace: Path,
) -> None:
    target = workspace / "target.txt"
    target.write_text("before", encoding="utf-8")

    class FailingJournal:
        def prepare(self, *args: object, **kwargs: object) -> object:
            raise OSError("journal unavailable")

    result = FileService(workspace, journal=FailingJournal()).write(
        "target.txt", "after", overwrite=True
    )

    assert result.status == "failed"
    assert result.recovery_class == "read_only"
    assert target.read_text(encoding="utf-8") == "before"


def test_target_drift_after_preimage_capture_is_blocked_without_overwrite(
    workspace: Path, tmp_path: Path
) -> None:
    target = workspace / "target.txt"
    target.write_text("before", encoding="utf-8")

    class RacingJournal(OperationJournal):
        def prepare(self, **kwargs):
            prepared = super().prepare(**kwargs)
            target.write_text("concurrent", encoding="utf-8")
            return prepared

    racing = RacingJournal(tmp_path / "racing" / "operations.sqlite3")
    result = FileService(workspace, journal=racing).write(
        "target.txt", "after", overwrite=True, operation_id="racing-op"
    )

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "conflict"
    assert target.read_text(encoding="utf-8") == "concurrent"


def test_recursive_delete_must_be_explicit(
    service: FileService, workspace: Path
) -> None:
    directory = workspace / "tree"
    directory.mkdir()
    (directory / "leaf.txt").write_text("leaf", encoding="utf-8")

    result = service.delete("tree", confirmed=True)

    assert result.status == "invalid_arguments"
    assert directory.exists()

