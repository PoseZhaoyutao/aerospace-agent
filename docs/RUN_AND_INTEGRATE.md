# 运行与 MCP Host 接入指南

## 1. 环境准备

### 1.1 Python 环境

```powershell
# Python ≥ 3.10
python --version

# 安装主 Agent 包
cd D:\Project\aerospace-agent
pip install -e .

# 安装 MCP Server 包
cd astro_dynamics_mcp
pip install -e .
```

### 1.2 引擎可选安装

```powershell
# 按需安装航天引擎（不装也能用，adapter 自动回退）
pip install orekit spiceypy astropy poliastro

# GMAT：从 https://sourceforge.net/projects/gmat/ 下载安装
# 设置环境变量：
$env:GMAT_PATH = "C:\GMAT\R2022a\bin\GMAT.exe"

# Basilisk：
pip install basilisksim

# STK（商业软件）：需安装 + 授权
# adapter 会通过 COM 自动检测
```

### 1.3 SPICE Kernel 准备

```powershell
# 下载必要 kernel 文件到 data/kernels/
# naif0012.tls — 闰秒
# de440s.bsp — 行星星历
# 下载地址：https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
```

## 2. 本地模型部署（可选）

### 2.1 Ollama（推荐）

```powershell
# 安装 Ollama：https://ollama.com
# 拉取模型
ollama pull qwen2.5:7b

# 设置环境变量
$env:AEROSPACE_LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"
$env:AEROSPACE_LOCAL_LLM_MODEL = "qwen2.5:7b"
```

### 2.2 vLLM

```powershell
pip install vllm
# 启动服务
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

$env:AEROSPACE_LOCAL_LLM_BASE_URL = "http://localhost:8000/v1"
$env:AEROSPACE_LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

### 2.3 云端 API（可选）

```powershell
$env:AEROSPACE_LLM_API_KEY = "sk-xxxx"
$env:AEROSPACE_LLM_BASE_URL = "https://api.openai.com/v1"
$env:AEROSPACE_LLM_MODEL = "gpt-4o-mini"
```

## 3. 运行 MCP Server

### 3.1 标准 MCP 协议模式

```powershell
# 需要 pip install mcp
python -m astro_dynamics_mcp.server
```

启动后会打印引擎可用性状态和已注册的 12 个工具。

### 3.2 命令行交互模式（无 mcp 包时回退）

```powershell
# 无需 mcp 包，直接交互测试工具
python -m astro_dynamics_mcp.server --cli
```

### 3.3 使用 Loop 引擎执行任务

```python
from aerospace_agent.core.llm_interface import create_llm
from astro_dynamics_mcp.loop import LoopEngine
from astro_dynamics_mcp.tools import TOOL_REGISTRY

# 创建 LLM（自动路由本地/云端）
llm = create_llm(use_router=True)

# 创建 Loop 引擎
engine = LoopEngine(llm=llm, tools=TOOL_REGISTRY)

# 执行任务
result = engine.execute(
    goal="设计地月转移轨道，精度<1km",
    constraints=["使用二体模型", "TLI速度增量<3.2km/s"],
)

print(f"状态: {result.status}")
print(f"引擎: {result.engine}")
print(f"验证: {result.validation}")
print(f"Loop 轮次: {len(result.loop_ledger)}")
```

## 4. 接入 Claude Desktop

编辑 Claude Desktop 配置文件：
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "astro-dynamics": {
      "command": "python",
      "args": ["-m", "astro_dynamics_mcp.server"],
      "env": {
        "GMAT_PATH": "C:\\GMAT\\R2022a\\bin\\GMAT.exe",
        "AEROSPACE_LOCAL_LLM_BASE_URL": "http://localhost:11434/v1",
        "AEROSPACE_LOCAL_LLM_MODEL": "qwen2.5:7b"
      }
    }
  }
}
```

重启 Claude Desktop 后，在对话中可直接使用航天动力学工具。

## 5. 接入 Cursor

编辑 Cursor 设置 → MCP Servers，添加：

```json
{
  "mcpServers": {
    "astro-dynamics": {
      "command": "python",
      "args": ["-m", "astro_dynamics_mcp.server"],
      "cwd": "D:\\Project\\aerospace-agent",
      "env": {
        "PYTHONPATH": "D:\\Project\\aerospace-agent\\astro_dynamics_mcp\\src"
      }
    }
  }
}
```

## 6. 接入 ChatGPT (OpenAI MCP Host)

在 ChatGPT 的 Custom Connectors 或通过 API 使用：

```python
import openai

client = openai.Client(api_key="sk-xxxx")

# 通过 MCP 工具调用航天动力学计算
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "计算北京站对 ISS 的可见性窗口"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "compute_ground_access",
                "description": "计算地面站对卫星的可见性窗口",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "orbit_state_dict": {"type": "object"},
                        "ground_station_dict": {"type": "object"},
                        "start_epoch_dict": {"type": "object"},
                        "stop_epoch_dict": {"type": "object"},
                        "min_elevation_deg": {"type": "number"}
                    }
                }
            }
        }
    ],
)
```

或使用 OpenAI 的 MCP Connector 配置指向本地 server。

## 7. 运行测试

```powershell
cd D:\Project\aerospace-agent\astro_dynamics_mcp

# 运行全部测试（无需安装任何引擎）
python -m pytest src/astro_dynamics_mcp/tests/ -v

# 运行特定测试
python -m pytest src/astro_dynamics_mcp/tests/test_time_tools.py -v
python -m pytest src/astro_dynamics_mcp/tests/test_propagation_tools.py -v
```

## 8. 快速验证

```powershell
# 验证包导入
python -c "from astro_dynamics_mcp import Epoch, Frame, OrbitState; print('Schema OK')"

# 验证适配器
python -c "from astro_dynamics_mcp.adapters import get_all_adapters; [print(a.info()) for a in get_all_adapters()]"

# 验证工具注册
python -c "from astro_dynamics_mcp.tools import TOOL_REGISTRY; print(f'{len(TOOL_REGISTRY)} tools registered')"

# 验证 Loop 引擎
python -c "from astro_dynamics_mcp.loop import LoopEngine; print('Loop engine OK')"

# 验证 LLM 接口（含本地模型）
python -c "from aerospace_agent.core.llm_interface import LocalLLM, ModelRouter; print('LLM interface OK')"
```

## 9. 环境变量汇总

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `AEROSPACE_LLM_API_KEY` | 云端 LLM API key | 无（用 MockLLM） |
| `AEROSPACE_LLM_BASE_URL` | 云端 LLM 基址 | https://api.openai.com/v1 |
| `AEROSPACE_LLM_MODEL` | 云端模型名 | gpt-3.5-turbo |
| `AEROSPACE_LOCAL_LLM_BASE_URL` | 本地模型端点 | http://localhost:11434/v1 |
| `AEROSPACE_LOCAL_LLM_MODEL` | 本地模型名 | qwen2.5:7b |
| `AEROSPACE_LOCAL_LLM_API_KEY` | 本地 API key（Ollama 不需要） | local-no-key-needed |
| `GMAT_PATH` | GMAT 可执行文件路径 | 无 |
| `OREKIT_DATA` | Orekit 数据目录 | ~/.orekitdata |
