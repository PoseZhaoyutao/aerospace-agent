import sys
from types import SimpleNamespace

from aerospace_agent.local_runtime import run_command
from aerospace_agent.utils import git_manager


def test_run_command_replaces_invalid_utf8_bytes_without_crashing():
    result = run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(bytes([0x80, 0x0a]))",
        ],
        timeout=5,
    )

    assert result.ok is True
    assert result.returncode == 0
    assert result.timeout is False
    assert "\ufffd" in result.stdout
    assert result.stderr == ""
    assert result.encoding == "utf-8"


def test_run_command_preserves_utf8_stdout():
    result = run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write('轨道传播'.encode('utf-8'))",
        ],
        timeout=5,
    )

    assert result.ok is True
    assert result.stdout == "轨道传播"
    assert result.stderr == ""


def test_run_command_timeout_is_structured():
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(3)"],
        timeout=0.1,
    )

    assert result.ok is False
    assert result.timeout is True
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


def test_git_manager_run_uses_safe_command_wrapper(monkeypatch):
    calls = []

    def fake_run_command(cmd, cwd=None, timeout=None, max_output_chars=None):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "timeout": timeout,
                "max_output_chars": max_output_chars,
            }
        )
        return SimpleNamespace(
            ok=False,
            returncode=1,
            stdout="stdout text",
            stderr="stderr text",
        )

    monkeypatch.setattr(git_manager, "run_command", fake_run_command)

    ok, output = git_manager.GitManager("repo-root")._run(["status"], timeout=7)

    assert ok is False
    assert output == "stderr text"
    assert calls == [
        {
            "cmd": ["git", "status"],
            "cwd": "repo-root",
            "timeout": 7,
            "max_output_chars": None,
        }
    ]
