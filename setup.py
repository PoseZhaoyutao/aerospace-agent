"""setuptools 打包配置。"""
from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


def _workspace_files(pattern: str) -> list[str]:
    """Return repository-relative runtime assets for sdist/wheel builds."""

    return sorted(path for path in glob(pattern, recursive=True) if Path(path).is_file())

setup(
    name="aerospace-agent",
    version="0.7.0",
    description="航天导航控制 Agent 核心框架",
    long_description="基于 ReAct 循环的航天动力学智能 Agent，集成：\n"
                     "  - CEO 三层上下文管理（Essential/Compress/Offload）\n"
                     "  - 三层记忆系统（短期/工作/长期 + MemoryManager）\n"
                     "  - 统一航天动力学 MCP（Canonical Model + 7 引擎适配器 + 12 工具）\n"
                     "  - Loop 八阶段自主交付循环\n"
                     "  - 可路由、可验证、可追踪的 RAG 知识系统\n"
                     "  - Skill 技能系统 + 任务专属 Prompt 模板\n"
                     "  - 可插拔 LLM（云端 OpenAI 兼容 / 本地 Ollama-vLLM / Mock 离线回退）",
    long_description_content_type="text/plain",
    author="Aerospace Agent Team",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(
        exclude=["astro_dynamics_mcp", "astro_dynamics_mcp.*",
                 "tests", "tests.*"],
    ),
    py_modules=["start_langgraph_agent"],
    include_package_data=True,
    package_data={
        "aerospace_agent.mcp": [
            "workflows/*.yaml",
            "examples/*.json",
            "prompts/*.md",
        ],
        "aerospace_agent.web": ["static/**/*"],
    },
    # Root-level runtime assets remain editable in development while also
    # being present in built distributions under a stable share directory.
    data_files=[
        ("share/aerospace-agent/config", _workspace_files("config/*.yaml")),
        ("share/aerospace-agent/knowledge", _workspace_files("knowledge/index.md") + _workspace_files("knowledge/log.md")),
        ("share/aerospace-agent/knowledge/orbital-dynamics", _workspace_files("knowledge/orbital-dynamics/*.md")),
        ("share/aerospace-agent/schemas/langgraph_agent", _workspace_files("schemas/langgraph_agent/*.json")),
        ("share/aerospace-agent/docs", _workspace_files("docs/LANGGRAPH_AGENT.md")),
    ],
    install_requires=[
        "numpy",
        "scipy",
        "click",
        "pyyaml",
        "langgraph>=1.0,<2.0",
        "langgraph-checkpoint-sqlite>=3.0,<4.0",
        "langchain-core>=1.0,<2.0",
        "pydantic>=2.0,<3.0",
        "mcp>=1.0,<2.0",
        "fastapi>=0.115,<1.0",
        "uvicorn[standard]>=0.30,<1.0",
    ],
    extras_require={
        "plot": ["matplotlib"],
        "rich": ["rich"],
        "mcp-server": ["mcp>=1.0,<2.0"],
        "local-llm": ["openai>=1.0.0"],
        "browser": ["playwright>=1.40,<2.0"],
        "engines": [
            "astropy",
            "poliastro",
            "spiceypy",
            "orekit",
        ],
        "dev": ["pytest", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "aerospace-agent=start_langgraph_agent:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
