from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "aerospace_agent" / "langgraph_agent"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_agent_facade_does_not_construct_concrete_domain_services():
    path = PACKAGE / "agent.py"
    source = path.read_text(encoding="utf-8")
    imports = _imports(path)

    assert "services.evolution" not in imports
    assert "EvolutionService(" not in source
    assert "KnowledgeService(" not in source
    assert "ContextService(" not in source
    assert "create_mcp_gateway(" not in source


def test_runtime_module_is_the_concrete_composition_root():
    runtime = (PACKAGE / "services" / "runtime.py").read_text(encoding="utf-8")
    assert "ContextService(" in runtime
    assert "KnowledgeService(" in runtime
    assert "EvolutionService(" in runtime
    assert "create_mcp_gateway(" in runtime

    for relative in ("agent.py", "nodes.py"):
        source = (PACKAGE / relative).read_text(encoding="utf-8")
        assert "create_mcp_gateway(" not in source


def test_persistent_rag_dependency_is_owned_by_this_workspace():
    import aerospace_agent.rag.aerospace_rag as rag_module
    import aerospace_agent.local_runtime as runtime_module

    for module in (rag_module, runtime_module):
        module_path = Path(module.__file__).resolve()
        assert module_path.is_relative_to(ROOT)


def test_mcp_tool_discovery_never_imports_adjacent_research_tools():
    tools_init = ROOT / "aerospace_agent" / "mcp" / "tools" / "__init__.py"
    source = tools_init.read_text(encoding="utf-8")

    assert "research_tools" not in source
    assert "_register_research_tools" not in source
