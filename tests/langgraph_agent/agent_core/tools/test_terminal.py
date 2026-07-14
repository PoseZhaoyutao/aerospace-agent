from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent_core.tools.terminal import TerminalService


@pytest.fixture
def service(tmp_path: Path) -> TerminalService:
    return TerminalService(
        tmp_path,
        allowed_commands=[sys.executable],
        env_allowlist=["SAFE_TEST_VALUE"],
    )


def test_rejects_shell_strings_metacharacters_and_non_allowlisted_commands(
    service: TerminalService,
) -> None:
    string_result = service.run(f'{sys.executable} --version')
    redirect_result = service.run([sys.executable, "--version", ">", "out.txt"])
    unavailable_result = service.run(["definitely-not-allowed", "--version"])

    assert string_result.status == "invalid_arguments"
    assert redirect_result.status == "invalid_arguments"
    assert unavailable_result.status == "unavailable"


def test_absolute_allowlist_does_not_authorize_same_basename_elsewhere(
    service: TerminalService, tmp_path: Path
) -> None:
    fake = tmp_path / "other" / Path(sys.executable).name
    fake.parent.mkdir()
    fake.write_bytes(b"not an executable")

    result = service.run([str(fake), "--version"])

    assert result.status == "unavailable"


def test_read_only_command_runs_without_confirmation(service: TerminalService) -> None:
    result = service.run([sys.executable, "--version"])

    assert result.status == "success"
    assert result.recovery_class == "read_only"
    assert result.result["returncode"] == 0


def test_unknown_or_writing_command_requires_confirmation_and_is_manual_recovery(
    service: TerminalService, tmp_path: Path
) -> None:
    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('written.txt').write_text('changed')",
    ]

    blocked = service.run(argv)
    allowed = service.run(argv, confirmed=True)

    assert blocked.status == "blocked"
    assert blocked.error is not None
    assert blocked.error.code == "confirmation_required"
    assert allowed.status == "success"
    assert allowed.recovery_class == "manual_recovery"
    assert (tmp_path / "written.txt").read_text(encoding="utf-8") == "changed"


def test_cwd_is_root_contained_and_environment_is_sanitized(
    service: TerminalService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHOULD_NOT_LEAK", "secret")
    child = tmp_path / "child"
    child.mkdir()
    script = (
        "import os; "
        "print(os.getcwd()); "
        "print(os.environ.get('SAFE_TEST_VALUE', '')); "
        "print(os.environ.get('SHOULD_NOT_LEAK', 'missing'))"
    )

    result = service.run(
        [sys.executable, "-c", script],
        cwd="child",
        env={"SAFE_TEST_VALUE": "safe"},
        confirmed=True,
    )
    outside = service.run(
        [sys.executable, "--version"], cwd=tmp_path.parent
    )
    bad_env = service.run(
        [sys.executable, "--version"], env={"SHOULD_NOT_LEAK": "bad"}
    )

    assert result.status == "success"
    assert result.result["stdout"].splitlines() == [str(child), "safe", "missing"]
    assert outside.status == "blocked"
    assert outside.error is not None
    assert outside.error.code == "path_outside_workspace"
    assert bad_env.status == "invalid_arguments"


def test_timeout_terminates_process_and_never_claims_reversibility(
    service: TerminalService,
) -> None:
    result = service.run(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_s=0.1,
        confirmed=True,
    )

    assert result.status == "timeout"
    assert result.error is not None
    assert result.error.code == "timeout"
    assert result.recovery_class == "manual_recovery"


def test_output_is_capped(service: TerminalService) -> None:
    result = service.run(
        [sys.executable, "-c", "print('x' * 200)"],
        max_output_chars=32,
        confirmed=True,
    )

    assert result.status == "success"
    assert result.result["truncated"] is True
    assert len(result.result["stdout"]) + len(result.result["stderr"]) <= 32


def test_background_status_and_confirmed_cancellation(service: TerminalService) -> None:
    started = service.run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        background=True,
        confirmed=True,
    )
    process_id = started.result["process_id"]

    running = service.status(process_id)
    blocked_cancel = service.cancel(process_id)
    cancelled = service.cancel(process_id, confirmed=True)

    assert started.status == "success"
    assert started.recovery_class == "manual_recovery"
    assert running.status == "success"
    assert running.result["running"] is True
    assert blocked_cancel.status == "blocked"
    assert cancelled.status == "success"
    assert cancelled.result["cancelled"] is True


def test_timeout_must_not_exceed_contract_limit(service: TerminalService) -> None:
    result = service.run([sys.executable, "--version"], timeout_s=121)

    assert result.status == "invalid_arguments"


@pytest.mark.parametrize(
    "outside_argument",
    ["absolute", "relative", "file_uri", "flag_value", "after_separator"],
)
def test_git_diff_rejects_workspace_escape_in_every_path_argument_form(
    tmp_path: Path, outside_argument: str
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required for terminal path-policy regression coverage")
    service = TerminalService(tmp_path, allowed_commands=[git])
    inside_a = tmp_path / "inside-a.txt"
    inside_b = tmp_path / "inside-b.txt"
    outside = tmp_path.parent / f"{tmp_path.name}-outside-secret.txt"
    inside_a.write_text("inside-a", encoding="utf-8")
    inside_b.write_text("inside-b", encoding="utf-8")
    outside.write_text("OUTSIDE_SECRET", encoding="utf-8")
    outside_relative = os.path.relpath(outside, tmp_path)

    arguments = {
        "absolute": [str(inside_a), str(outside)],
        "relative": [str(inside_a), outside_relative],
        "file_uri": [str(inside_a), outside.as_uri()],
        "flag_value": [f"--output={outside}", str(inside_a), str(inside_b)],
        "after_separator": ["--", str(inside_a), outside_relative],
    }[outside_argument]

    result = service.run([git, "diff", "--no-index", *arguments])

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "path_outside_workspace"
    assert "OUTSIDE_SECRET" not in str(result.result)

