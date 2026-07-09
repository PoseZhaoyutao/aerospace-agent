"""Local runtime helpers for safe command execution."""

from .process import CommandResult, run_command

__all__ = ["CommandResult", "run_command"]
