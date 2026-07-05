"""setuptools 打包配置。"""
from setuptools import find_packages, setup

setup(
    name="aerospace-agent",
    version="0.1.0",
    description="航天导航控制 Agent 核心框架",
    long_description="基于 ReAct 循环的航天导航控制 Agent，集成可插拔 LLM、"
                     "CEO 上下文管理、记忆系统、工具与工作流编排。",
    long_description_content_type="text/plain",
    author="Aerospace Agent Team",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "click",
    ],
    extras_require={
        "plot": ["matplotlib"],
        "rich": ["rich"],
    },
    entry_points={
        "console_scripts": [
            "aerospace-agent=aerospace_agent.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
