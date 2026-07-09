"""File-based skill manifest discovery and validation.

This module treats local ``SKILL.md`` files as declarative capabilities. It
does not execute them. Executable Python skills remain owned by
``SkillRegistry.register`` and ``SkillBase``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEPENDENCY_KEYS = {"dependencies", "requires"}


@dataclass
class SkillManifest:
    name: str
    description: str
    path: str
    root: str
    category: str = "external"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    available: bool = True
    issues: List[str] = field(default_factory=list)
    executable: bool = False
    source: str = "file"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _split_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text]


def _split_front_matter(text: str) -> tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_index: Optional[int] = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text

    return _parse_front_matter(lines[1:end_index]), "\n".join(lines[end_index + 1:])


def _parse_front_matter(lines: Iterable[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in lines:
        if not raw_line.strip():
            continue
        stripped = raw_line.strip()
        if stripped.startswith("-") and current_key:
            value = _strip_quotes(stripped[1:].strip())
            existing = metadata.setdefault(current_key, [])
            if isinstance(existing, list):
                existing.append(value)
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not value:
            metadata[key] = []
            current_key = key
        elif key in DEPENDENCY_KEYS:
            metadata[key] = _split_list(value)
            current_key = key
        else:
            metadata[key] = value
            current_key = key
    return metadata


def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _first_paragraph(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return ""


def _dependency_module_name(dependency: str) -> str:
    for marker in ("==", ">=", "<=", "~=", ">", "<", "["):
        if marker in dependency:
            dependency = dependency.split(marker, 1)[0]
    return dependency.strip().replace("-", "_")


def validate_skill_manifest(path: str | Path, root: str | Path | None = None) -> Dict[str, Any]:
    manifest_path = Path(path)
    issues: List[str] = []
    metadata: Dict[str, Any] = {}
    body = ""

    if not manifest_path.is_file():
        return SkillManifest(
            name=manifest_path.parent.name,
            description="",
            path=str(manifest_path),
            root=str(Path(root) if root else manifest_path.parent),
            available=False,
            issues=["missing:SKILL.md"],
        ).to_dict()

    try:
        text = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = manifest_path.read_text(encoding="utf-8", errors="replace")
        issues.append("decode_replaced")

    metadata, body = _split_front_matter(text)
    name = str(metadata.get("name") or _first_heading(body) or manifest_path.parent.name).strip()
    description = str(metadata.get("description") or _first_paragraph(body)).strip()
    category = str(metadata.get("category") or "external").strip()

    dependencies: List[str] = []
    for key in DEPENDENCY_KEYS:
        dependencies.extend(_split_list(metadata.get(key)))
    dependencies = list(dict.fromkeys(dependencies))

    if not name:
        issues.append("missing:name")
    if not description:
        issues.append("missing:description")

    for dependency in dependencies:
        module_name = _dependency_module_name(dependency)
        if not module_name or importlib.util.find_spec(module_name) is None:
            issues.append(f"missing_dependency:{dependency}")

    return SkillManifest(
        name=name,
        description=description,
        category=category,
        path=str(manifest_path),
        root=str(Path(root) if root else manifest_path.parent),
        dependencies=dependencies,
        metadata=metadata,
        available=not issues,
        issues=issues,
    ).to_dict()


def discover_skill_manifests(roots: Iterable[str | Path]) -> List[Dict[str, Any]]:
    manifests: List[Dict[str, Any]] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue

        candidates: List[Path] = []
        root_manifest = root / "SKILL.md"
        if root_manifest.is_file():
            candidates.append(root_manifest)
        for child in sorted(root.iterdir()):
            if child.is_dir():
                candidate = child / "SKILL.md"
                if candidate.is_file():
                    candidates.append(candidate)

        for candidate in candidates:
            manifests.append(validate_skill_manifest(candidate, root=root))
    return manifests
