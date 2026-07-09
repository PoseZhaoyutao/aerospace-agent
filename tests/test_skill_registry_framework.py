from pathlib import Path

from aerospace_agent.skills.manifest import validate_skill_manifest
from aerospace_agent.skills.registry import SkillRegistry


def test_skill_registry_discovers_file_skill_manifests_without_executing(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "orbit-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: orbit-review
description: Reviews orbit propagation assumptions.
category: analysis
dependencies: json, definitely_missing_aerospace_dep_zz
---

# Orbit Review

Use this to audit orbit propagation assumptions.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry(skill_roots=[skill_root])

    assert registry.discover_manifests() == 1
    manifests = registry.list_skill_manifests()
    assert len(manifests) == 1
    assert manifests[0]["name"] == "orbit-review"
    assert manifests[0]["category"] == "analysis"
    assert manifests[0]["executable"] is False
    assert manifests[0]["path"].endswith(str(Path("orbit-review") / "SKILL.md"))

    result = registry.execute(None, "orbit-review")
    assert result["success"] is False
    assert result["error_code"] == "SKILL_NOT_EXECUTABLE"


def test_validate_skill_manifest_reports_missing_dependencies(tmp_path):
    skill_dir = tmp_path / "sensor-skill"
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    path.write_text(
        """---
name: sensor-skill
description: Validates sensor simulation context.
dependencies: json, definitely_missing_aerospace_dep_zz
---

# Sensor Skill
""",
        encoding="utf-8",
    )

    result = validate_skill_manifest(path)

    assert result["name"] == "sensor-skill"
    assert result["available"] is False
    assert "missing_dependency:definitely_missing_aerospace_dep_zz" in result["issues"]


def test_skill_manifest_parser_falls_back_to_heading_and_first_paragraph(tmp_path):
    skill_dir = tmp_path / "starfield"
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    path.write_text(
        """# Starfield QA

Checks that star catalog, camera, PSF, and truth metadata are explicit.
""",
        encoding="utf-8",
    )

    result = validate_skill_manifest(path)

    assert result["name"] == "Starfield QA"
    assert result["description"] == (
        "Checks that star catalog, camera, PSF, and truth metadata are explicit."
    )
    assert result["available"] is True
