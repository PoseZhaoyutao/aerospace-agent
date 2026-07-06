# GitHub 推送指南

把沙盒中的航天导航控制 Agent 推送到你自己的 GitHub 仓库。

## 方法一：一键脚本（推荐）

### 步骤 1：在 GitHub 创建空仓库

1. 登录 https://github.com
2. 点击右上角 "+" -> "New repository"
3. 仓库名填 `aerospace-agent`
4. **不要勾选** "Initialize this repository with a README"
5. 点击 "Create repository"
6. 复制仓库地址（HTTPS）：`https://github.com/你的用户名/aerospace-agent.git`

### 步骤 2：配置远程并推送

在沙盒终端执行：

```bash
cd /workspace

# 添加远程仓库（替换为你的用户名）
git remote add origin https://github.com/你的用户名/aerospace-agent.git

# 推送所有分支和标签
git push -u origin main
```

如果提示输入用户名和密码：
- 用户名：你的 GitHub 用户名
- 密码：使用 GitHub Personal Access Token（不是登录密码）
  - 生成方法：https://github.com/settings/tokens -> Generate new token (classic)
  - 勾选 `repo` 权限
  - 复制 token 作为密码输入

推送成功后，你的完整代码（含 git 历史）就在 GitHub 上了。

---

## 方法二：下载 Git Bundle（离线迁移）

如果沙盒无法连接 GitHub，使用 bundle 文件：

### 在沙盒中导出

```bash
cd /workspace
git bundle create aerospace_agent.bundle HEAD
```

bundle 文件已生成在 `/workspace/aerospace_agent.bundle`。

### 在 Windows 本地恢复

1. 把 `aerospace_agent.bundle` 复制到 Windows 电脑（如 `D:\Downloads`）
2. 打开 PowerShell 或 CMD：

```powershell
cd D:\
git clone D:\Downloads\aerospace_agent.bundle D:\Project\aerospace-agent
cd D:\Project\aerospace-agent
```

这样你就得到了完整的 git 仓库（含所有提交历史）。

---

## 方法三：直接 ZIP 下载（最简单，无 git 历史）

```powershell
# 在 Windows 中
Expand-Archive -Path "D:\Downloads\aerospace_agent.zip" -DestinationPath "D:\Project\aerospace-agent"
```

---

## Windows 本地环境配置

### 1. 安装 Python 3.10+

https://www.python.org/downloads/

勾选 "Add Python to PATH"

### 2. 安装依赖

```powershell
cd D:\Project\aerospace-agent
python -m pip install --upgrade pip
pip install numpy scipy matplotlib click

# 可选：安装航天专业库（增强 MCP 工具为真实模式）
pip install orekit spiceypy astropy Basilisk
```

### 3. 配置 LLM API（可选）

创建 `.env` 文件：

```
AEROSPACE_LLM_API_KEY=sk-your-key
AEROSPACE_LLM_BASE_URL=https://api.openai.com/v1
AEROSPACE_LLM_MODEL=gpt-4o
```

不配置则自动使用 MockLLM（离线可用）。

### 4. 运行

```powershell
# 安装为命令行工具
pip install -e .

# 运行地月转移演示
aerospace-agent demo

# 搜索最新文献
aerospace-agent rag literature "lunar transfer orbit" --report

# 查看全部命令
aerospace-agent --help
```

---

## 项目结构速览

```
aerospace-agent/
├── aerospace_agent/
│   ├── core/              # Agent 内核：LLM接口、CEO上下文管理、记忆、ReAct编排
│   ├── physics/           # 物理引擎：开普勒、二体、Lambert、拼凑圆锥、地月转移
│   ├── mcp_tools/         # 6个航天工具：orekit/gmat/spiceypy/astropy/basilisk/stk
│   ├── rag/               # RAG系统：向量库+BM25+知识图谱+文献管线+知识云图
│   ├── workflows/         # 4个工作流：轨道设计/发射窗口/地月转移/Basilisk可视化
│   ├── reporting/         # 报告生成：7张图+HTML报告+知识学习报告
│   ├── utils/             # Git管理
│   ├── cli.py             # Click 命令行
│   └── demo.py            # 端到端演示编排
├── data/                  # 持久化数据（记忆、RAG索引、知识图谱、下载的PDF）
├── reports/               # 生成的 HTML 报告
├── demo_outputs/          # 生成的图表
├── setup.py               # Python 打包
├── requirements.txt       # 依赖清单
└── README.md              # 项目说明
```
