# Herta — 基于 RAG 的工业协议处理器代码自动生成系统

> **本仓库内说明**：这是 herta 引擎的原始文档。在 domain-codegen 平台中 herta 仅作为 Python 引擎被 `server/` 编排调用（生成器、评测、知识库数据），其自带的 `frontend/` 演示界面与 `output/reports/`、`output/visualization/` 历史报告已移除——平台统一使用根目录 `app/` 前端。下文涉及 frontend/reports 的段落仅作引擎原始形态参考。

从已有协议文档与代码构建知识库，结合 LLM 检索增强生成（RAG）自动生成目标协议的配置文件与 C 处理代码，并提供多维度自动化评测。

> 场景：卫星等无人平台的通信协议文档（Word）繁杂琐碎，人工将非结构化协议文档转写为 C 星务控制代码耗时易错。Herta 将这一过程流水线化：**协议文档 → 结构化中间表示（JSON Config）→ 协议处理代码（Handler C）→ 自动评测**。

## 功能概览

| 模块 | 说明 |
|------|------|
| 知识库构建 | 解析协议文档（docx/pdf）、C 源码、JSON 配置，代码结构感知分块 + FAISS 向量索引 |
| Config 生成 | 基于目标协议文档，多 query 检索 + 中文 bigram 重排序，生成结构化 JSON 配置 |
| Handler 生成 | Plan → Code → Verification 三阶段策略生成协议处理 C 代码 |
| 评测对比 | JSON 四维加权对比 + tree-sitter AST 树编辑距离 + 匈牙利算法函数匹配 |
| Web 前端 | Next.js 蓝白科技风界面：仪表盘、知识库管理、代码生成、评测对比、设置 |
| Embedding 可视化 | UMAP 2D 投影展示知识库向量空间分布 |

## 快速开始

### 环境要求

- Python ≥ 3.12
- Node.js ≥ 18（含 npm）
- [智谱 AI BigModel 开放平台](https://open.bigmodel.cn/) API Key

### 1. 克隆并安装后端

```bash
git clone https://github.com/fangzheng649/HERTA.git
cd HERTA

# 方式 A：conda
conda create -n herta python=3.12 -y
conda activate herta

# 方式 B：任意 Python 3.12 虚拟环境均可

pip install -e .
# 或 pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd frontend
npm install --legacy-peer-deps
cd ..
```

### 3. 配置 LLM 供应商

支持两类供应商，**对话（Chat）与向量（Embedding）可分别配置**：

| 供应商 | 协议 | 说明 |
|--------|------|------|
| `zai`（默认） | 智谱 AI 原生（zai-sdk） | 对话 + embedding-3 向量 |
| `openai` | OpenAI 兼容 HTTP | DeepSeek、硅基流动、Moonshot 等 |

```bash
# 方式一：智谱 AI（默认，一把钥匙同时用于对话与向量）
export ZAI_API_KEY='your-zhipu-key'

# 方式二：DeepSeek 对话 + 智谱向量（DeepSeek 无 embedding 接口）
export ZAI_API_KEY='your-zhipu-key'                 # embedding 仍走智谱
export HERTA_CHAT_PROVIDER=openai
export HERTA_CHAT_BASE_URL='https://api.deepseek.com'
export HERTA_CHAT_API_KEY='sk-your-deepseek-key'
export HERTA_CHAT_MODEL='deepseek-chat'
```

也可以在 Web 界面 **设置** 页面可视化配置（仅当前进程有效，重启需重设）。

> ⚠️ 注意：DeepSeek 官方 API 仅提供对话模型，**不提供 embedding 接口**。如需全链路 OpenAI 兼容，可改用硅基流动等同时提供对话与向量（如 `BAAI/bge-m3`）的服务商。

### 4. 启动服务

```bash
# 终端 1：启动后端 API（默认 127.0.0.1:8000）
herta api

# 终端 2：启动前端开发服务器（默认 3000）
cd frontend && npm run dev
```

浏览器访问 **http://localhost:3000** 即可使用 Web 界面。前端通过 `next.config.mjs` 将 `/api/*` 代理到后端 8000 端口。

> 仓库已附带一份预构建的 `gdwx` 知识库与 `dllg` 评测结果，克隆后打开仪表盘即可看到数据，无需先跑完整流水线。

### 5. （可选）体验完整流水线

```bash
# 重新构建知识库（调用 embedding API，约几分钟）
herta buildkb --source gdwx --force

# 生成目标协议 dllg 的配置与处理代码（调用 LLM）
herta generate config  --target dllg
herta generate handler --target dllg

# 评测对比（与 ground truth 比较，本地计算无需 API）
herta compare --target dllg
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `herta api` | 启动 Web API 服务（`--host` / `--port` / `--reload`） |
| `herta buildkb --source <名称>` | 构建知识库（`--chunk-strategy` 可选 simple/semantic/autosize/proposition/code_aware/direct） |
| `herta generate config --target <名称>` | 生成结构化配置（`--top-k`、`--model`、`--temperature`） |
| `herta generate handler --target <名称>` | 三阶段生成处理代码（`--save-plan` 保留中间设计草图） |
| `herta compare --target <名称>` | 运行评测，产出 `output/reports/<目标>/metrics.json` 与 Markdown 报告 |

## Web 界面

| 页面 | 功能 |
|------|------|
| 仪表盘 | 知识库概览、最新评测指标卡、Config 分维图表、系统信息 |
| 知识库 | 知识库列表、在线构建/重建、Embedding 2D 散点图 |
| 代码生成 | 选择目标协议与知识库，生成 Config / Handler 并查看代码 |
| 评测对比 | 七维评分卡、相似度图表、函数级匹配明细表 |
| 设置 | API Key、Chat / Embedding 模型配置 |

## 评测指标

**Config 相似度**（四维加权）：

```
Config = 0.20 × 拓扑结构 + 0.30 × 命令表 + 0.30 × 遥测处理 + 0.20 × 内部处理
```

**Handler 相似度**：tree-sitter 解析 C AST，APTED 树编辑距离（权重 0.70）+ 六类语义特征 Jaccard 相似度（权重 0.30），函数级用匈牙利算法做最优一对一匹配。

**整体**：`Overall = 0.35 × Config + 0.65 × Handler`

仓库附带一轮完整评测结果（gdwx → dllg）：

| 指标 | 得分 |
|------|------|
| 拓扑结构 | 80.0% |
| 命令表 | 25.6% |
| 遥测处理 | 94.4% |
| 内部处理 | 100.0% |
| **Config 综合** | **72.0%** |
| **Handler 综合** | **48.0%** |
| **整体相似度** | **56.4%** |

## 项目结构

```
HERTA/
├── src/herta/                  # 核心代码
│   ├── api/                    # FastAPI Web API（routes/ + 异步任务）
│   ├── cli/                    # Typer CLI（api / buildkb / generate / compare）
│   ├── kb/                     # 知识库（分块策略、FAISS 向量存储）
│   ├── generator/              # 生成（Config / Handler / 检索器 / 目标文档解析）
│   ├── compare/                # 评测（JSON 对比、AST 相似度、函数匹配、报告）
│   ├── parser/                 # 解析（docx/pdf 文档、C/JSON 代码）
│   └── llm/                    # 智谱 AI 客户端
├── frontend/                   # Next.js 15 前端（App Router + Tailwind + VChart）
├── prompts/                    # Prompt 模板（config / handler plan / handler code）
├── data/
│   ├── source_proto/gdwx/      # 源协议（文档 + 代码 + 配置）
│   ├── target_proto/dllg/      # 目标协议（Ground Truth）
│   └── kb/gdwx/                # 预构建知识库（FAISS 索引 + chunks）
├── output/                     # 产物（generated / reports / visualization）
├── tools/                      # 辅助脚本（Embedding 可视化导出）
└── tests/smoke_test.py         # LLM 客户端冒烟测试（需 API Key）
```

## 技术栈

**后端**：Python 3.12 · FastAPI · Typer · FAISS · tree-sitter · APTED · RapidFuzz · SciPy · 智谱 AI（BigModel）

**前端**：Next.js 15 · React 19 · TypeScript · Tailwind CSS · VChart (VisActor)

**RAG 链路**：多 query 向量召回 + 中文 bigram 重排序 + 结构优先级上下文组装 + 三阶段生成与后验证

## 文档

- [架构设计](doc/architecture.md)
- [CLI 使用说明](doc/cli.md)
- [相似度计算详解](doc/similarity_calculation.md)
- [系统介绍](doc/system_introduction.md)

## 环境变量

| 变量 | 必填 | 说明 | 默认 |
|------|------|------|------|
| `ZAI_API_KEY` | 任一密钥必填 | 智谱 AI API Key（chat/embedding 兜底） | — |
| `HERTA_CHAT_PROVIDER` | 否 | 对话供应商：`zai` / `openai` | `zai` |
| `HERTA_CHAT_API_KEY` | 否 | 对话密钥（缺省回退 ZAI_API_KEY） | — |
| `HERTA_CHAT_BASE_URL` | openai 时必填 | 对话接口地址，如 `https://api.deepseek.com` | — |
| `HERTA_CHAT_MODEL` | 否 | 对话模型 | `glm-5.1` |
| `HERTA_EMBEDDING_PROVIDER` | 否 | 向量供应商：`zai` / `openai` | `zai` |
| `HERTA_EMBEDDING_API_KEY` | 否 | 向量密钥（缺省回退 ZAI_API_KEY） | — |
| `HERTA_EMBEDDING_BASE_URL` | openai 时必填 | 向量接口地址 | — |
| `HERTA_EMBEDDING_MODEL` | 否 | 向量模型 | `embedding-3` |
| `HERTA_EMBEDDING_DIMENSIONS` | 否 | 向量维度 | `1024` |
| `HERTA_DATA_DIR` | 否 | 数据目录 | `data` |
| `HERTA_OUTPUT_DIR` | 否 | 输出目录 | `output` |

## 常见问题

**Q：没有 API Key 能看什么？**
可以。仓库附带预构建的 `gdwx` 知识库与 `dllg` 完整评测结果，启动后无需任何 Key 即可浏览：仪表盘指标、知识库 Embedding 可视化、已生成的 Config/Handler 代码、评测对比与完整 Markdown 报告。只有「构建知识库」「生成 Config/Handler」需要 LLM API；「运行评测」为纯本地计算。

**Q：端口被占用？**
后端换端口：`herta api --port 8001`，同时把 `frontend/next.config.mjs` 中的 `localhost:8000` 改为对应端口。前端换端口：`npm run dev -- -p 3001`。

**Q：Windows 下 `herta api` 提示找不到命令？**
确认已 `pip install -e .` 且当前环境的 Scripts 目录在 PATH 中；也可用等价命令 `python -m uvicorn herta.api.app:app --port 8000 --app-dir src`。

**Q：构建知识库卡住或失败？**
多为 embedding 接口问题：检查 API Key 是否有效、embedding 供应商配置是否正确（DeepSeek 无 embedding 接口，见上文供应商配置）。

## License

MIT
