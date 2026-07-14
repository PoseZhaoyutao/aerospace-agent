import ast
from pathlib import Path


def _setup_tree():
    return ast.parse(Path("setup.py").read_text(encoding="utf-8"))


def test_setup_declares_webui_dependencies_and_static_package_data():
    tree = _setup_tree()
    install_requires = {
        item.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "install_requires"
        for item in node.value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    assert "fastapi>=0.115,<1.0" in install_requires
    assert "uvicorn[standard]>=0.30,<1.0" in install_requires
    source = Path("setup.py").read_text(encoding="utf-8")
    assert '"aerospace_agent.web": ["static/**/*"]' in source


def test_built_webui_index_exists_after_frontend_build():
    index = Path("aerospace_agent/web/static/index.html")
    if not index.is_file():
        import pytest

        pytest.skip("WEBUI_FRONTEND_BUILD_UNAVAILABLE")
    assert index.stat().st_size > 0
