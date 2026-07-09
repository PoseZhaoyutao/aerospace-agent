"""Default local skill roots for the aerospace agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional


def _unique_existing(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        result.append(resolved)
    return result


def default_skill_roots(
    home: Optional[Path] = None,
    env_value: Optional[str] = None,
) -> List[Path]:
    """Return existing local roots for default declarative skills.

    This does not download anything. It discovers local Codex/superpowers skill
    directories that already exist on the machine.
    """

    if env_value is None:
        env_value = os.environ.get("AEROSPACE_DEFAULT_SKILL_ROOTS", "")
    configured = [
        Path(part)
        for part in env_value.split(os.pathsep)
        if part.strip()
    ]

    home = home or Path.home()
    codex_home = Path(os.environ.get("CODEX_HOME", home / ".codex"))
    candidates = configured + [
        codex_home / "superpowers" / "skills",
        codex_home / "skills" / "pdf",
        codex_home / "skills" / ".system" / "pdf",
    ]

    plugin_root = codex_home / "plugins" / "cache" / "openai-primary-runtime"
    if plugin_root.exists():
        candidates.extend(plugin_root.glob("*/*/skills"))
        candidates.extend(plugin_root.glob("*/skills"))

    return _unique_existing(candidates)


def install_default_skill_manifests(registry, roots: Optional[Iterable[str | Path]] = None) -> int:
    """Attach default skill roots to a registry and discover SKILL.md files."""

    resolved_roots = [Path(root).resolve() for root in (roots or default_skill_roots())]
    for root in resolved_roots:
        if root not in getattr(registry, "skill_roots", []):
            registry.skill_roots.append(root)
    if not resolved_roots:
        return 0
    return registry.discover_manifests(resolved_roots)
