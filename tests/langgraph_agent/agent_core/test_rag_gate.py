from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from aerospace_agent.langgraph_agent.agent_core.rag_gate import (
    ExecutionRunStore,
    RagGateDecision,
    RagGateService,
    decide_private_rag,
)


def test_private_rag_uses_only_three_positive_triggers_and_negative_override() -> None:
    assert decide_private_rag(route="knowledge_qa", confidence=0.2, user_text="解释轨道摄动").retrieve
    assert decide_private_rag(route="knowledge_qa", confidence=0.9, user_text="请给出依据和来源").retrieve
    assert decide_private_rag(
        route="complex_task",
        confidence=0.9,
        user_text="规划任务",
        planner_request="retrieve",
    ).retrieve

    assert not decide_private_rag(route="knowledge_qa", confidence=0.9, user_text="解释轨道摄动").retrieve
    assert not decide_private_rag(route="conversation", confidence=0.1, user_text="你好").retrieve
    assert not decide_private_rag(
        route="conversation",
        confidence=0.1,
        user_text="你好",
        planner_request="retrieve",
    ).retrieve
    assert not decide_private_rag(
        route="conversation",
        confidence=0.1,
        user_text="你好",
        planner_request="retrieve",
    ).retrieve
    denied = decide_private_rag(
        route="knowledge_qa",
        confidence=0.1,
        user_text="不要核实，不需要来源",
        planner_request="retrieve",
    )
    assert denied == RagGateDecision(retrieve=False, reason="explicit_no_retrieval")


def test_user_and_scheduled_runs_have_correct_initial_budget(tmp_path) -> None:
    store = ExecutionRunStore(tmp_path / "execution_runs.sqlite")
    user = store.create_user_run(
        root_run_id="run-user",
        project_id="project",
        thread_id="thread",
    )
    scheduled = store.create_scheduled_run(
        root_run_id="job:1:1",
        project_id="project",
    )

    assert user.retrieval_budget == 1 and user.retrieval_state == "available"
    assert scheduled.retrieval_budget == 0 and scheduled.retrieval_state == "unavailable"
    assert store.schema_version() == 1


def test_atomic_claim_allows_only_one_parallel_branch(tmp_path) -> None:
    store = ExecutionRunStore(tmp_path / "execution_runs.sqlite")
    run = store.create_user_run(
        root_run_id="run",
        project_id="project",
        thread_id="thread",
    )

    def claim(index: int):
        return store.claim(
            root_run_id="run",
            expected_version=run.version,
            claimer_id=f"branch-{index}",
            query="orbit propagation",
            reason="low_confidence",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(claim, range(4)))

    claimed = [item for item in results if item is not None]
    assert len(claimed) == 1
    assert claimed[0].retrieval_state == "claimed"


def test_request_is_marked_in_flight_before_retriever_is_called_and_consumed_after(tmp_path) -> None:
    store = ExecutionRunStore(tmp_path / "execution_runs.sqlite")
    run = store.create_user_run(
        root_run_id="run",
        project_id="project",
        thread_id="thread",
    )
    observations: list[str] = []

    def retriever(query: str):
        observations.append(store.get("run").retrieval_state)
        return [{"text": f"evidence for {query}"}]

    result = RagGateService(store).retrieve_once(
        run=run,
        decision=RagGateDecision(retrieve=True, reason="explicit_evidence_request"),
        query="orbit",
        claimer_id="branch",
        retriever=retriever,
    )

    assert result == [{"text": "evidence for orbit"}]
    assert observations == ["in_flight"]
    assert store.get("run").retrieval_state == "consumed"
    assert store.claim(
        root_run_id="run",
        expected_version=store.get("run").version,
        claimer_id="again",
        query="orbit",
        reason="retry",
    ) is None


def test_no_gate_decision_never_calls_retriever(tmp_path) -> None:
    store = ExecutionRunStore(tmp_path / "execution_runs.sqlite")
    run = store.create_user_run(root_run_id="run", project_id="project", thread_id="thread")
    calls: list[str] = []

    result = RagGateService(store).retrieve_once(
        run=run,
        decision=RagGateDecision(retrieve=False, reason="high_confidence"),
        query="orbit",
        claimer_id="branch",
        retriever=lambda query: calls.append(query),
    )

    assert result is None
    assert calls == []
    assert store.get("run").retrieval_state == "available"


def test_retriever_failure_still_consumes_budget(tmp_path) -> None:
    store = ExecutionRunStore(tmp_path / "execution_runs.sqlite")
    run = store.create_user_run(root_run_id="run", project_id="project", thread_id="thread")

    def fail(_: str):
        raise TimeoutError("private RAG timed out")

    try:
        RagGateService(store).retrieve_once(
            run=run,
            decision=RagGateDecision(retrieve=True, reason="low_confidence"),
            query="orbit",
            claimer_id="branch",
            retriever=fail,
        )
    except TimeoutError:
        pass

    assert store.get("run").retrieval_state == "consumed"


def test_crash_recovery_releases_unstarted_claim_but_never_retries_in_flight(tmp_path) -> None:
    current = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    store = ExecutionRunStore(tmp_path / "execution_runs.sqlite", clock=lambda: current[0])
    first = store.create_user_run(root_run_id="first", project_id="project", thread_id="thread")
    second = store.create_user_run(root_run_id="second", project_id="project", thread_id="thread")
    first_claim = store.claim(
        root_run_id="first",
        expected_version=first.version,
        claimer_id="branch",
        query="one",
        reason="low_confidence",
        lease_seconds=10,
    )
    second_claim = store.claim(
        root_run_id="second",
        expected_version=second.version,
        claimer_id="branch",
        query="two",
        reason="low_confidence",
        lease_seconds=10,
    )
    assert first_claim is not None and second_claim is not None
    store.mark_in_flight(
        root_run_id="second",
        expected_version=second_claim.version,
        claimer_id="branch",
    )
    current[0] += timedelta(seconds=11)

    recovered = store.recover_expired()

    assert recovered == {"released": 1, "consumed_unknown": 1}
    assert store.get("first").retrieval_state == "available"
    assert store.get("second").retrieval_state == "consumed_unknown"
    assert store.claim(
        root_run_id="second",
        expected_version=store.get("second").version,
        claimer_id="retry",
        query="two",
        reason="retry",
    ) is None

