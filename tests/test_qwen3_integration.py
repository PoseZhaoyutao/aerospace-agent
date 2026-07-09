# -*- coding: utf-8 -*-
"""
Qwen3 + AerospaceAgent 完整 ReAct 循环集成测试。

使用本地部署的 Qwen3-VL-8B-Instruct 替代 MockLLM，
验证 Agent 能通过 ReAct 循环完成真实的航天计算任务。

测试场景:
  1. 工具调用：计算 400km 圆轨道速度（触发 orbital_velocity 工具）
  2. 数学计算：用 calculator 工具计算轨道周期
  3. RAG 检索：查询轨道力学知识
"""
import sys
import os
import time

sys.path.insert(0, r"d:\Project\aerospace-agent")
os.environ["AEROSPACE_LOCAL_LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
os.environ["AEROSPACE_LOCAL_LLM_MODEL"] = "qwen3-vl"

from aerospace_agent.core.llm_interface import LocalLLM
from aerospace_agent.core.agent import create_default_agent, AerospaceAgent


def create_qwen3_agent(max_steps: int = 8) -> AerospaceAgent:
    """创建使用 Qwen3 的 Agent（非 MockLLM）。"""
    llm = LocalLLM(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen3-vl",
        max_retries=3,
        retry_delay=2.0,
    )
    agent = create_default_agent(force_mock=False, max_steps=max_steps)
    # 替换 LLM 为 Qwen3
    agent.llm = llm
    return agent


def test_tool_call():
    """场景 1：Agent 通过 ReAct 调用 orbital_velocity 工具。"""
    print("\n" + "=" * 70)
    print("场景 1：工具调用 — 计算 400km 圆轨道速度")
    print("=" * 70)

    agent = create_qwen3_agent(max_steps=6)
    task = "请使用 orbital_velocity 工具计算 400km 高度圆轨道的轨道速度。先调用工具，再根据结果给出 Final Answer。"

    print(f"\n任务: {task}")
    print(f"LLM: {agent.llm.base_url} / {agent.llm.model}")
    print(f"可用工具: {list(agent.tools.keys()) + list(agent.mcp_tools.keys())}")
    print()

    t0 = time.time()
    result = agent.run(task)
    elapsed = time.time() - t0

    print(f"\n结果: {result}")
    print(f"耗时: {elapsed:.1f}s")

    # 验证：结果应包含速度数值（约 7.67 km/s）
    success = "7.6" in str(result) or "7.7" in str(result)
    print(f"\n{'✓ 通过' if success else '✗ 需检查'} — 预期 ~7.67 km/s")
    return success


def test_calculator():
    """场景 2：Agent 通过 ReAct 调用 calculator 工具。"""
    print("\n" + "=" * 70)
    print("场景 2：数学计算 — 用 calculator 工具计算 2*pi*sqrt(6778137^3/398600441800000)")
    print("=" * 70)

    agent = create_qwen3_agent(max_steps=6)
    task = "请使用 calculator 工具计算表达式 2*math.pi*math.sqrt(6778137**3/398600441800000) ，然后给出 Final Answer。"

    print(f"\n任务: {task}")
    print()

    t0 = time.time()
    result = agent.run(task)
    elapsed = time.time() - t0

    print(f"\n结果: {result}")
    print(f"耗时: {elapsed:.1f}s")

    # 验证：结果应包含 ~5553s（轨道周期）或 ~1.54h
    success = "5553" in str(result) or "5554" in str(result) or "1.5" in str(result)
    print(f"\n{'✓ 通过' if success else '✗ 需检查'} — 预期 ~5553s 或 ~1.54h")
    return success


def test_rag_query():
    """场景 3：RAG 知识检索。"""
    print("\n" + "=" * 70)
    print("场景 3：RAG 知识检索 — 查询霍曼转移轨道")
    print("=" * 70)

    agent = create_qwen3_agent(max_steps=4)

    if agent.rag is None:
        print("✗ 跳过 — RAG 不可用")
        return False

    t0 = time.time()
    # 直接测试 RAG 检索
    result = agent.rag.query("霍曼转移轨道", top_k=3)
    elapsed = time.time() - t0

    print(f"\nRAG 检索结果:\n{result[:500]}")
    print(f"\n耗时: {elapsed:.1f}s")

    success = "霍曼" in result or "Hohmann" in result.lower() or "转移" in result
    print(f"\n{'✓ 通过' if success else '✗ 需检查'}")
    return success


def test_full_react_with_rag():
    """场景 4：完整 ReAct 循环 — Agent 结合工具和知识回答问题。"""
    print("\n" + "=" * 70)
    print("场景 4：完整 ReAct — 计算地月转移 TLI 速度增量")
    print("=" * 70)

    agent = create_qwen3_agent(max_steps=8)
    task = (
        "计算从 400km 停泊轨道到月球转移轨道(TLI)的速度增量。"
        "请先用 orbit_calculator 工具计算，然后给出 Final Answer。"
    )

    print(f"\n任务: {task}")
    print()

    t0 = time.time()
    result = agent.run(task)
    elapsed = time.time() - t0

    print(f"\n结果: {result}")
    print(f"耗时: {elapsed:.1f}s")

    # TLI 速度增量约 3.1-3.2 km/s
    success = "3.0" in str(result) or "3.1" in str(result) or "3.2" in str(result)
    print(f"\n{'✓ 通过' if success else '✗ 需检查'} — 预期 ~3.1 km/s")
    return success


if __name__ == "__main__":
    print("=" * 70)
    print("  Qwen3 + AerospaceAgent 完整集成测试")
    print("  LLM: Qwen3-VL-8B-Instruct @ http://127.0.0.1:8000")
    print("=" * 70)

    results = {}

    # 场景 1：工具调用
    try:
        results["tool_call"] = test_tool_call()
    except Exception as e:
        print(f"\n异常: {e}")
        results["tool_call"] = False

    # 场景 2：数学计算
    try:
        results["calculator"] = test_calculator()
    except Exception as e:
        print(f"\n异常: {e}")
        results["calculator"] = False

    # 场景 3：RAG 检索
    try:
        results["rag_query"] = test_rag_query()
    except Exception as e:
        print(f"\n异常: {e}")
        results["rag_query"] = False

    # 场景 4：完整 ReAct
    try:
        results["full_react"] = test_full_react_with_rag()
    except Exception as e:
        print(f"\n异常: {e}")
        results["full_react"] = False

    # 汇总
    print("\n" + "=" * 70)
    print("  测试汇总")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name:20s} {status}")

    total = sum(results.values())
    print(f"\n  通过率: {total}/{len(results)}")
    print("=" * 70)
