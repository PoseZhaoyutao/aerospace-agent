from aerospace_agent.core.context_manager import ContextManager


def test_context_manager_snapshot_tracks_sources_and_layers(tmp_path):
    manager = ContextManager(offload_dir=str(tmp_path / "offload"))

    manager.add_essential("Original mission requirement.", source="user", kind="task_spec")
    manager.add_message("user", "Propagate one LEO orbit.", source="conversation", kind="prompt")
    manager.add_tool_record(
        "space.propagate_orbit",
        {"duration_s": 60},
        {"status": "completed"},
        source="mcp",
        kind="tool_result",
    )
    manager.save_offload(
        "state_history",
        {"samples": [{"elapsed_s": 0}, {"elapsed_s": 60}]},
        source="experiment",
        kind="artifact",
    )

    snapshot = manager.snapshot()

    assert snapshot["counts"]["essential"] == 1
    assert snapshot["counts"]["messages"] == 1
    assert snapshot["counts"]["tool_records"] == 1
    assert snapshot["counts"]["offload"] == 1
    assert snapshot["layers"]["essential"]["items"][0]["source"] == "user"
    assert snapshot["layers"]["essential"]["items"][0]["kind"] == "task_spec"
    assert snapshot["layers"]["compress"]["messages"][0]["source"] == "conversation"
    assert snapshot["layers"]["compress"]["tool_records"][0]["source"] == "mcp"
    assert snapshot["layers"]["offload"]["items"][0]["source"] == "experiment"
    assert snapshot["estimated_tokens"]["total"] >= 1

    rendered = manager.build_context()
    assert "Original mission requirement." in rendered
    assert "[user] Propagate one LEO orbit." in rendered


def test_context_manager_keeps_backward_compatible_add_methods(tmp_path):
    manager = ContextManager(offload_dir=str(tmp_path / "offload"))

    manager.add_essential("Keep this exact task.")
    manager.add_message("assistant", "Acknowledged.")
    manager.add_tool_record("tool.name", {"x": 1}, {"ok": True})

    snapshot = manager.snapshot()

    assert snapshot["layers"]["essential"]["items"][0]["source"] == "unknown"
    assert snapshot["layers"]["compress"]["messages"][0]["kind"] == "message"
    assert snapshot["layers"]["compress"]["tool_records"][0]["kind"] == "tool_record"
