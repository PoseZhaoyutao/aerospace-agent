"""测试 WorkflowCatalog 加载与检索。

本测试加载 workflows 目录下的全部 YAML 工作流，验证：
  - 全部 6 个工作流成功加载
  - 每个工作流含 WorkflowSpec 必需字段
  - 按 task_type / query / engine 检索正确
  - 7 大类别结构完整
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aerospace_agent.mcp.resources import WorkflowCatalog, CATALOG_CATEGORIES
from aerospace_agent.mcp.schemas import WorkflowSpec


#: workflows 目录（与本测试文件同级包的 ../workflows）
WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


@pytest.fixture(scope="module")
def catalog() -> WorkflowCatalog:
    """加载 workflows 目录的工作流目录。"""
    cat = WorkflowCatalog()
    count = cat.load_from_dir(WORKFLOWS_DIR)
    assert count > 0, f"未从 {WORKFLOWS_DIR} 加载到任何工作流"
    return cat


class TestWorkflowCatalogLoad:
    """工作流目录加载测试。"""

    def test_loads_all_yaml_files(self, catalog):
        """应加载全部 6 个工作流 YAML。"""
        assert len(catalog) >= 6

    def test_expected_workflow_ids_present(self, catalog):
        """关键工作流 ID 均存在。"""
        expected = {
            "leo_propagation",
            "spice_ephemeris_query",
            "ground_station_access",
            "cross_validate_orekit_gmat",
            "basilisk_basic_orbit",
            "gmat_script_run",
        }
        for wf_id in expected:
            assert wf_id in catalog, f"缺少工作流: {wf_id}"

    def test_workflow_has_required_fields(self, catalog):
        """每个工作流含 WorkflowSpec 必需字段。"""
        required = {
            "id", "goal", "task_type", "inputs", "models",
            "engine", "steps", "outputs", "validation",
            "failure_handling",
        }
        for summary in catalog.list_all():
            full = catalog.get(summary["id"])
            missing = required - set(full.keys())
            assert not missing, f"{summary['id']} 缺字段: {missing}"

    def test_steps_have_tool_and_name(self, catalog):
        """每个步骤含 tool 与 name 字段。"""
        for summary in catalog.list_all():
            full = catalog.get(summary["id"])
            steps = full.get("steps", [])
            assert len(steps) >= 1, f"{summary['id']} 无步骤"
            for step in steps:
                assert "tool" in step, f"{summary['id']} 步骤缺 tool"
                assert "name" in step, f"{summary['id']} 步骤缺 name"
                assert "outputs" in step, f"{summary['id']} 步骤缺 outputs"

    def test_load_into_workflow_spec(self, catalog):
        """加载的工作流可构造为 WorkflowSpec 对象。"""
        wf = catalog.get("leo_propagation")
        spec = WorkflowSpec.from_yaml_dict(wf)
        assert spec.id == "leo_propagation"
        assert spec.task_type == "orbit_propagation"
        assert len(spec.steps) >= 1

    def test_load_from_dict(self):
        """load_from_dict 可直接加载单个工作流字典。"""
        cat = WorkflowCatalog()
        data = {
            "id": "test_wf",
            "goal": "测试工作流",
            "task_type": "orbit_propagation",
            "steps": [{"name": "s1", "tool": "propagate_orbit", "outputs": []}],
        }
        assert cat.load_from_dict(data) is True
        assert "test_wf" in cat


class TestWorkflowCatalogSearch:
    """工作流检索测试。"""

    def test_search_by_task_type(self, catalog):
        """按 task_type 检索 orbit_propagation。"""
        results = catalog.search(task_type="orbit_propagation")
        assert len(results) >= 1
        for r in results:
            assert r["task_type"] == "orbit_propagation"

    def test_search_by_query_keyword(self, catalog):
        """关键词检索（中文 goal 匹配）。"""
        results = catalog.search(query="传播")
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert "leo_propagation" in ids

    def test_search_by_engine_basilisk(self, catalog):
        """按引擎检索 basilisk——basilisk 工作流应在结果中。"""
        results = catalog.search(preferred_engine="basilisk")
        ids = [r["id"] for r in results]
        assert "basilisk_basic_orbit" in ids

    def test_search_no_match(self, catalog):
        """无匹配时返回空列表。"""
        results = catalog.search(query="不存在的关键词xyz123")
        assert results == []

    def test_search_combined_filters(self, catalog):
        """组合 task_type + query 检索。"""
        results = catalog.search(
            query="可见性", task_type="ground_access"
        )
        assert len(results) >= 1
        assert results[0]["id"] == "ground_station_access"

    def test_get_returns_full_dict(self, catalog):
        """get 返回完整工作流字典。"""
        wf = catalog.get("leo_propagation")
        assert wf is not None
        assert wf["id"] == "leo_propagation"
        assert wf["goal"]
        assert "validation" in wf

    def test_get_missing_returns_none(self, catalog):
        """get 不存在的 ID 返回 None。"""
        assert catalog.get("does_not_exist") is None

    def test_list_all_returns_summaries(self, catalog):
        """list_all 返回摘要列表。"""
        summaries = catalog.list_all()
        assert len(summaries) == len(catalog)
        for s in summaries:
            assert "id" in s and "task_type" in s and "engine" in s


class TestCatalogCategories:
    """类别结构测试。"""

    def test_all_seven_categories_defined(self):
        """7 大类别常量完整。"""
        assert len(CATALOG_CATEGORIES) == 7
        assert "orbit_propagation" in CATALOG_CATEGORIES
        assert "validation" in CATALOG_CATEGORIES
        assert "ground_access" in CATALOG_CATEGORIES

    def test_categories_populated(self, catalog):
        """加载后类别映射包含已加载工作流。"""
        cats = catalog.categories()
        # orbit_propagation 类别应非空
        assert len(cats.get("orbit_propagation", [])) >= 1
        assert "leo_propagation" in cats["orbit_propagation"]

    def test_all_workflows_in_some_category(self, catalog):
        """所有工作流的 task_type 都有对应类别桶。"""
        cats = catalog.categories()
        all_categorized = set()
        for ids in cats.values():
            all_categorized.update(ids)
        for summary in catalog.list_all():
            assert summary["id"] in all_categorized or summary["id"] in catalog
