# architecture.md

````markdown
# Herta 项目架构设计文档

## 1. 文档目的

本文档用于定义 Herta 实验系统的整体架构、模块边界、数据流、目录约定和实现原则，供 Cursor 后续自动生成代码时作为统一设计依据。

Herta 的目标是构建一个本地 CLI 软件系统，基于源协议 `gdwx` 的协议文档与配套代码建立本地知识库，然后读取目标协议 `dllg` 的协议文档，自动调用 LLM API 生成对应的：

- `dllg_config.json`
- `dllg_handler.c`

最后再由系统对生成结果和真实代码进行逻辑相似度分析并输出报告。

---

## 2. 系统目标

### 2.1 功能目标

系统应支持以下核心能力：

1. 从源协议文档和代码构建本地知识库。
2. 从目标协议文档中抽取协议要素。
3. 使用本地知识库和 LLM 生成目标协议配置与处理代码。
4. 将生成结果与真实代码进行结构和逻辑相似度比较。
5. 通过统一 CLI 命令完成全流程实验。

### 2.2 非功能目标

系统应满足：

- **可复现**：同样输入和配置下可重复运行。
- **可扩展**：未来可支持更多协议对。
- **可解释**：生成和对比过程应保留中间产物和报告。
- **可维护**：模块职责清晰，便于 Cursor 分阶段生成代码。
- **可替换**：知识库引擎、LLM 客户端、对比策略应可替换。

---

## 3. 项目目录设计

基于当前项目目录，建议约定如下：

```text
目录树
herta/
├── data/                                #用于存放源协议、目标协议和知识库数据
│   ├── source_proto/
│   │   └── gdwx/
│   │       ├── gdwx_config.json         # 源协议 gdwx 的配置文件，作为知识库构建输入之一
│   │       ├── cmd_handler.c            # 源协议中的遥控指令处理代码，用于提取命令处理逻辑
│   │       ├── gdwx_handler.c           # 源协议中的遥测处理代码，用于提取帧解析与上报逻辑
│   │       ├── internal_handler.c       # 源协议中的内部通信处理代码，补充协议内部处理细节
│   │       └── gdwx.docx                # 源协议文档，提供协议结构、字段定义和说明信息
│   ├── target_proto/
│   │   └── dllg/
│   │       ├── dllg_config.json         # 目标协议的真实配置文件，用于对比评测 （必用）
│   │       ├── cmd_handler.c            # 目标协议的真实遥控处理代码，可选用于对照分析 （可选）
│   │       ├── dllg_handler.c           # 目标协议的真实处理代码，作为生成结果的对比基准 （必用）
│   │       ├── dllg.doc                 # 目标协议文档，是生成配置和代码的 核心输入 （必用）
│   │       └── internal_handler.c       # 目标协议的内部通信处理代码，可用于补充对比逻辑 （可选）
│   └── kb/
│       └── gdwx/
│           ├── chunks.jsonl             # 知识库切分后的文本与代码块数据
│           ├── meta.json                # 知识库元数据，如来源文件、分块信息、构建配置等
│           ├── vector.index             # 向量索引文件，用于相似语义检索
│           └── embeddings.npy           # 分块后的 embedding 向量数据
│
├── doc/                                 # 用于存放项目设计文档。
│   ├── architecture.md                  # 项目架构设计文档，描述系统模块、流程和目录约定
│   └── cli.md                           # CLI 命令说明文档，定义 herta 命令行接口规范
│
├── prompts/                             # 用于存放各阶段使用的 Prompt 模板
│   ├── generate_config.md               # 生成目标协议配置文件时使用的 Prompt 模板
│   ├── generate_handler_plan.md           # 生成 handler 设计草图（阶段A）的 Prompt 模板
│   ├── generate_handler_code.md           # 生成 handler 完整代码（阶段B）的 Prompt 模板
│   ├── extract_protocol.md                # 【预留】从协议文档中抽取结构化信息时使用的 Prompt 模板
│   └── compare_logic.md                   # 【预留】对生成结果做逻辑比较和评测时使用的 Prompt 模板
│
├── src/
│   └── herta/                           # 系统核心实现目录
│       ├── __init__.py                  # 包初始化文件
│       ├── main.py                      # 项目主入口，可用于程序启动或调试入口
│       ├── config.py                    # 系统配置加载与管理模块
│       ├── logging.py                   # 日志配置模块，统一定义日志输出方式
│       ├── constants.py                 # 常量定义模块，集中管理固定配置项和默认值
│       ├── cli/
│       │   ├── __init__.py              # CLI 子包初始化文件
│       │   ├── app.py                   # CLI 根应用入口，注册所有子命令
│       │   ├── buildkb.py               # buildkb 子命令实现，负责构建知识库
│       │   ├── generate.py              # generate 子命令实现，负责生成目标协议代码
│       │   ├── compare.py               # compare 子命令实现，负责结果对比评测
│       │   ├── runall.py                # 【预留】runall 子命令实现，负责一键执行全流程
│       │   └── inspect.py               # 【预留】inspect 子命令实现，负责查看知识库或中间产物
│       ├── parser/
│       │   ├── __init__.py              # parser 子包初始化文件
│       │   ├── doc_parser.py            # 协议文档解析模块，负责读取 doc/docx/pdf 并抽取结构
│       │   ├── code_parser.py           # 代码解析模块，负责切分 C/JSON 文件并提取结构信息
│       │   └── protocol_schema.py       # 【预留】协议统一中间表示定义模块
│       ├── kb/                          # 知识库（KB/RAG）模块
│       │   ├── __init__.py              # kb 子包初始化文件
│       │   ├── schemas.py               # KB 数据模型定义（SourceDocument, Chunk, KBMeta）
│       │   ├── chunker.py               # 分块策略基类模块，定义 BaseChunker 抽象接口
│       │   ├── simple_chunker.py        # 简单分块策略：固定大小滑动窗口
│       │   ├── semantic_chunker.py      # 语义分块策略：基于相邻句子相似度切分
│       │   ├── selector.py              # 自动分块大小选择策略（Auto-Size）
│       │   ├── proposition_chunker.py   # 命题分块策略：基于原子事实命题
│       │   ├── builder.py               # 知识库构建模块，负责组织分块、embedding 和索引
│       │   └── store.py                 # 向量存储抽象模块，封装 FAISS 索引实现
│       ├── generator/                     # 生成编排层（实现为独立子包）
│       │   ├── __init__.py              # generator 子包初始化文件
│       │   ├── schemas.py               # 生成相关数据模型（TargetProtocolSpec, RetrievedChunk等）
│       │   ├── prompt_loader.py         # Prompt 模板加载与渲染模块
│       │   ├── target_parser.py         # 目标协议文档解析模块
│       │   ├── retriever.py             # 知识库检索封装模块
│       │   ├── config_generator.py      # 配置生成器：生成 config.json
│       │   └── handler_generator.py     # Handler 生成器：两阶段生成 handler.c
│       ├── llm/
│       │   ├── __init__.py              # llm 子包初始化文件
│       │   └── client.py                # LLM 客户端封装，负责模型 API 调用（chat + embeddings）
│       ├── compare/                       # 对比评测层
│       │   ├── __init__.py              # compare 子包初始化文件
│       │   ├── json_compare.py          # JSON 配置对比模块，计算 5 个维度相似度
│       │   ├── c_ast_parser.py          # C 代码 AST 解析模块（tree-sitter）
│       │   ├── ast_similarity.py        # AST 相似度计算（TED + 语义特征）
│       │   ├── function_matcher.py      # 函数匹配模块（匈牙利算法）
│       │   └── report_builder.py        # 报告生成模块（Markdown + JSON）
│       └── utils/
│           ├── __init__.py              # utils 子包初始化文件
│           ├── fileio.py                # 文件读写工具函数
│           ├── paths.py                 # 路径处理工具函数
│           ├── text.py                  # 文本处理工具函数
│           └── hash.py                  # 【预留】哈希工具函数
│
├── output/                              # 用于存放生成结果、中间产物和评测报告
│   ├── generated/
│   │   └── dllg/
│   │       ├── dllg_config.json         # 生成得到的目标协议配置文件
│   │       └── dllg_handler.c           # 生成得到的目标协议处理代码
│   ├── intermediate/
│   │   └── dllg/
│   │       ├── parsed_doc.json          # 目标协议文档解析后的结构化结果
│   │       ├── retrieved_context.json   # 生成时从知识库检索得到的上下文内容
│   │       ├── prompt_config.txt        # 生成配置文件时实际发送给模型的 Prompt
│   │       └── prompt_handler.txt       # 生成处理代码时实际发送给模型的 Prompt
│   └── reports/
│       └── dllg/
│           ├── comparison_report.md     # 对比评测报告，面向阅读和分析
│           └── metrics.json             # 对比评测指标结果，便于程序处理和统计
│
├── requirements.txt                     # Python 依赖列表，适用于基础安装方式
├── pyproject.toml                       # 项目构建配置与 CLI 入口定义文件
└── README.md                            # 项目说明文档，介绍用途、安装和使用方式
````

说明：

* CLI 入口统一封装为可执行命令 `herta`，而不是直接调用 `cli.py`。
* `output/` 中分离生成结果、中间产物和评测报告，便于追踪实验过程。

---

## 4. 总体架构

### 4.1 架构分层

Herta 采用五层架构：

1. **CLI 接入层**：接收命令、参数校验、任务调度。
2. **协议理解层**：解析文档和代码，抽取结构化协议信息。
3. **知识库检索层**：构建源协议知识库并提供检索。
4. **生成编排层**：加载 Prompt，调用 LLM，生成配置与代码。
5. **评测报告层**：对生成结果与真实结果做结构和语义逻辑比较。

### 4.2 核心流程

#### 流程 A：构建知识库（RAG Ingestion）

输入：

- `gdwx.docx`
- `gdwx_config.json`
- `cmd_handler.c`
- `gdwx_handler.c`
- `internal_handler.c`

处理：

1. 文档加载与文本提取。
2. 文档预处理（清洗、标准化为json数据类型、按文件类型识别结构）。
3. 根据配置选择分块策略：
   - 简单分块（固定 chunk_size + overlap）
   - 语义分块（基于句间/段间语义边界）
   - 分块大小选择（自动评估候选参数并选择最佳配置）
   - 命题分块（将文本拆解为原子事实命题）
4. 为每个 chunk 生成 metadata。
5. 计算 embedding。
6. 建立本地向量索引与元数据存储。
7. 输出构建报告，记录分块策略、参数、chunk 数量和索引状态。

输出：

- `data/kb/gdwx/`
- `data/kb/gdwx/chunks.jsonl`
- `data/kb/gdwx/meta.json`
- `data/kb/gdwx/vector.index`
- `data/kb/gdwx/embeddings.npy`
- `data/kb/gdwx/build_report.json`

#### 流程 B：基于目标协议生成代码

输入：

* `dllg.doc`
* `data/kb/gdwx/`

处理：

1. 解析目标协议文档, 转换成 json 文件 ，便于处理。
2. 抽取命令、帧、字段、中间结构。
3. 按生成任务检索相关上下文。
4. 先生成 `dllg_config.json`。
5. 再基于配置和检索结果生成 `dllg_handler.c`。
6. 保存中间 Prompt 和生成结果。

输出：

* `output/generated/dllg/dllg_config.json`
* `output/generated/dllg/dllg_handler.c`

#### 流程 C：结果对比评测

输入：

* 生成的 `dllg_config.json`
* 真实的 `dllg_config.json`
* 生成的 `dllg_handler.c`
* 真实的 `dllg_handler.c`

处理：

1. JSON 结构与映射比较。
2. C 代码结构比较。
3. 语义逻辑事实抽取。
4. 综合评分。
5. 输出 Markdown 和 JSON 报告。

输出：

* `output/reports/dllg/comparison_report.md`
* `output/reports/dllg/metrics.json`

---

## 5. 模块设计

### 5.1 CLI 模块

位置：`src/herta/cli/`

职责：

* 定义 `herta` 根命令。
* 注册子命令：`buildkb`、`generate`、`compare`、`runall`、`inspect`。
* 统一参数解析、日志输出和异常退出码。

建议框架：

* `Typer` 或 `Click`

设计原则：

* 子命令短小、语义明确。
* 支持默认路径，减少手工输入。
* 所有命令支持 `--verbose` 和 `--dry-run` 扩展位。

---

### 5.2 文档解析模块

位置：`src/herta/parser/doc_parser.py`

职责：

* 解析 `.docx`、`.doc` 文档。
* 抽取纯文本和章节结构。
* 将协议文档转换为统一中间结构。

输出目标结构示例：

```json
{
  "protocol_name": "dllg",
  "commands": [
    {
      "cmd_name": "查询状态",
      "cmd_id": "0x01",
      "request_fields": [],
      "response_fields": []
    }
  ],
  "telemetry_frames": [],
  "internal_format": {}
}
```

设计原则：

* 优先规则抽取。
* 规则不足时允许使用 LLM 做文档结构化摘要。
* 保留原始文本和结构化结果，便于排查。

---

### 5.3 代码解析模块

位置：`src/herta/parser/code_parser.py`

职责：

* 读取 `.c`、`.json` 文件。
* 将代码按函数、case、配置项切块。
* 为知识库构建和逻辑抽取提供基础结构。

处理对象：

* `*_config.json`
* `cmd_handler.c`
* `*_handler.c`
* `internal_handler.c`

建议能力：

* 基础实现：正则 + 规则分段。
* 进阶实现：引入 `tree-sitter-c` 做 AST 级结构分析。

---

### 5.4 协议中间表示模块

位置：`src/herta/parser/protocol_schema.py`

职责：

* 定义协议统一中间表示。
* 约束文档解析输出、生成输入和评测输入。

建议定义的数据对象：

* `ProtocolSpec`
* `CommandSpec`
* `TelemetryFrameSpec`
* `FieldSpec`
* `InternalFormatSpec`

设计要求：

* 统一源协议和目标协议表示。
* 字段尽量可序列化为 JSON。

---

### 5.5 知识库模块（RAG Ingestion & Indexing）

位置：`src/herta/kb/`

职责：

- 执行 RAG 准备阶段中的文档摄取、分块、向量化和索引构建。
- 保存 chunk、metadata、embedding、索引文件。
- 为后续检索和生成提供统一知识库接口。

子模块：

- `chunker.py`：统一分块入口，按策略路由到不同分块实现。
- `simple_chunker.py`：简单分块（固定大小 + overlap）。
- `semantic_chunker.py`：语义分块。
- `selector.py`：分块大小选择器，自动评估候选参数。
- `proposition_chunker.py`：命题分块。
- `builder.py`：构建知识库主流程。
- `embedder.py`：embedding 计算封装。
- `retriever.py`：检索接口。
- `store.py`：向量存储抽象。

#### 支持的分块策略

##### 1. 简单分块（simple）

基于固定 `chunk_size` 和 `chunk_overlap` 切分文本。

适用场景：

- 快速原型验证
- 文本结构不稳定的文档
- 对构建速度要求较高的场景

特点：

- 实现简单
- 构建速度快
- 可能打断语义边界

##### 2. 语义分块（semantic）

先按句子或段落切分，再根据相邻单元的语义相似度决定分块边界。

适用场景：

- 协议说明文档
- 章节结构清晰的技术文档
- 需要提高块内主题一致性的场景

特点：

- 块内语义更完整
- 检索质量通常优于固定分块
- 构建成本高于简单分块

##### 3. 分块大小选择（auto-size）

对多组候选 `chunk_size` / `chunk_overlap` 参数进行自动评估，选择最优配置后再正式构建知识库。

适用场景：

- 不确定最佳分块参数
- 希望针对协议领域自动优化分块配置
- 需要可重复评估的实验型场景

特点：

- 参数选择更科学
- 构建耗时更长
- 依赖评估样本或问题集

##### 4. 命题分块（proposition）

通过 LLM 或规则+LLM 方法，将文档拆解为最小事实命题，每个 chunk 对应一个或一组原子事实。

适用场景：

- 协议字段定义
- 配置映射关系
- 需要精确事实检索与可追溯性的场景

特点：

- 粒度最细
- 检索精度高
- 构建成本最高
- 对 LLM 质量和输出稳定性有要求

#### 统一 chunk 数据结构建议

```json
{
  "chunk_id": "gdwx_000123",
  "source": "gdwx",
  "file": "gdwx.docx",
  "section": "遥测协议定义",
  "chunk_strategy": "semantic",
  "chunk_index": 12,
  "text": "...",
  "token_count": 318,
  "char_count": 842,
  "proposition_count": null,
  "parent": {
    "doc_id": "gdwx.docx",
    "page": 4,
    "paragraph": 18
  }
}
```

#### metadata 建议字段

```json
{
  "protocol": "gdwx",
  "file": "gdwx_handler.c",
  "kind": "code",
  "section": "telemetry_handler",
  "command_id": "0x81",
  "chunk_strategy": "simple",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

#### 构建产物建议

* `chunks.jsonl`：保存所有 chunk 及其 metadata
* `meta.json`：保存构建参数、源文件列表、分块策略、embedding 模型等
* `vector.index`：向量索引
* `embeddings.npy`：原始向量数据
* `build_report.json`：保存构建统计信息，例如：

  * 文件数
  * chunk 总数
  * 平均 chunk 长度
  * 分块策略
  * 参数配置
  * 是否启用自动选择
  * 构建耗时

#### 设计原则

1. `builder.py` 不直接依赖某一种分块方法，而是通过统一策略接口调用。
2. 所有分块方法都输出统一 chunk 结构，便于后续检索与调试。
3. 文档与代码允许使用不同默认分块策略，但统一由 `buildkb` 配置驱动。
4. 命题分块属于增强模式，默认关闭。
5. 先完成分块接入，再逐步扩展重排序、上下文压缩等后续能力。

---

#### 5.5.1 分块策略接口设计

建议为所有分块器定义统一接口：

```python
class BaseChunker:
    def chunk(self, text: str, metadata: dict) -> list[dict]:
        ...
```

不同分块器实现：

* `SimpleChunker`
* `SemanticChunker`
* `AutoSizeChunker`
* `PropositionChunker`

统一输入：

* 原始文本
* 文件 metadata
* 分块配置参数

统一输出：

* chunk 列表，每个 chunk 均包含：

  * `text`
  * `chunk_id`
  * `chunk_index`
  * `chunk_strategy`
  * `metadata`

这样 `buildkb` 只需要决定“选哪种 chunker + 传什么参数”，不需要感知具体实现细节。

### 5.6 Generator 生成编排模块

位置：`src/herta/generator/`

职责：

* 解析目标协议文档，提取结构化信息。
* 从知识库检索相关上下文。
* 加载和渲染 Prompt 模板。
* 编排多阶段代码生成流程（配置生成、Handler 两阶段生成）。
* 校验生成结果格式并保存。

子模块：

* `schemas.py`：生成相关数据模型定义（TargetProtocolSpec, RetrievedChunk, GenerationContext 等）。
* `target_parser.py`：目标协议文档解析，提取摘要和章节结构。
* `retriever.py`：知识库检索封装，支持向量相似度搜索。
* `prompt_loader.py`：Prompt 模板加载与变量渲染。
* `config_generator.py`：配置生成器，生成目标协议的 config.json。
* `handler_generator.py`：Handler 生成器，采用两阶段生成策略（设计草图 + 完整代码）。

#### 两阶段 Handler 生成流程

**阶段A - 生成设计草图（Plan）**：

输入：
* 目标协议文档结构化结果
* 从知识库检索到的 handler 代码片段和文档

处理：
1. 调用 LLM 生成设计草图
2. 包含函数列表、命令分发逻辑、遥测处理逻辑、内部通信逻辑
3. 明确标注 TODO 和未知点

输出：
* 结构化设计计划（Markdown 格式，可选保存到 `output/intermediate/`）

**阶段B - 生成完整代码（Code）**：

输入：
* 阶段A生成的设计草图
* 检索到的参考上下文
* Prompt 模板

处理：
1. 基于设计草图调用 LLM 生成完整 C 代码
2. 确保代码结构清晰，包含必要的错误处理
3. 添加 TODO 注释标记不确定的部分

输出：
* 完整的 handler.c 代码文件

设计要求：

* 使用低温度（temperature=0.1），保证生成结果可复现。
* 保存完整的 Prompt 和模型响应，便于追溯。
* 对输出进行基础格式校验（JSON 有效性、C 代码完整性检查）。
* 未知字段用 `null` 或 `"TODO"` 占位，禁止虚构。

---

### 5.6.1 LLM 客户端模块

位置：`src/herta/llm/`

职责：

* 封装智谱 AI (BigModel) API 调用。
* 统一处理 Chat Completions 和 Embeddings 请求。
* 管理 API 密钥和模型配置。
* 实现批量处理和自动分批（适配 API 限制）。

子模块：

* `client.py`：LLM 客户端封装，提供 `chat()` 和 `embed_texts()` 接口。

接口示例：

```python
# 聊天调用
client = BigModelClient()
response = client.chat(
    messages=[{"role": "user", "content": "..."}],
    model="glm-5.1",
    temperature=0.1
)

# Embedding 调用（自动分批处理）
embeddings = client.embed_texts(
    texts=["text1", "text2", ...],
    model="embedding-3",
    batch_size=64  # 自动分批
)
```

---

### 5.7 对比评测模块

位置：`src/herta/compare/`

状态：**已实现**

职责：

* 比较 LLM 生成的 `config.json` 与真实配置文件的相似度。
* 比较生成的 `handler.c` 与真实代码的 AST 级相似度。
* 使用匈牙利算法进行函数一对一匹配。
* 输出 `metrics.json` 结构化指标和 `comparison_report.md` 可读性报告。

子模块：

* `__init__.py`：模块导出
* `json_compare.py`：JSON 配置对比，计算 5 个维度相似度
* `c_ast_parser.py`：C 代码 AST 解析（tree-sitter）
* `ast_similarity.py`：AST 相似度计算（tree-edit distance + 语义特征）
* `function_matcher.py`：函数匹配（匈牙利算法）
* `report_builder.py`：报告生成（Markdown + JSON）

#### JSON 配置对比维度

1. **topology_similarity** (20% 权重)
   - 检查 4 层树结构：根节点 → 6 个顶层字段 → 3 个列表 → 列表元素字段
   - 验证字段存在性和类型正确性

2. **cmd_list_similarity** (30% 权重)
   - 按 `cmd` 对齐命令项
   - 比较 `handler` 精确匹配和 `desc` 模糊匹配（rapidfuzz）

3. **tele_handler_similarity** (30% 权重)
   - 按 `handler` 对齐遥测项
   - 比较 `len`、`desc` 字段

4. **internal_handler_similarity** (20% 权重)
   - 按 `handler` 对齐内部处理项
   - 比较 `len`、`desc` 字段

5. **config_similarity** (加权总分)
   - 公式：0.2*topology + 0.3*cmd_list + 0.3*tele_handler + 0.2*internal

#### C 代码 AST 对比流程

1. **AST 解析**：使用 tree-sitter 提取函数定义（函数名、返回类型、参数、函数体）
2. **相似度计算**：
   - tree_edit_similarity (70%)：使用 apted 算法计算树编辑距离
   - semantic_feature_similarity (30%)：比较调用函数、字面量、赋值模式、标识符等
3. **函数匹配**：使用 scipy.optimize.linear_sum_assignment（匈牙利算法）求解最优一对一匹配
4. **整体评分**：基于平均相似度、覆盖率、同名匹配率计算 handler_similarity

5. **overall_similarity** (整体相似度)
   - 公式：0.35*config_similarity + 0.65*handler_similarity

#### 输出文件

* `metrics.json`：结构化指标，包含所有相似度分数和详细对比结果
* `comparison_report.md`：可读性报告，包含评分表格、问题列表、改进建议

依赖：
- `tree-sitter` + `tree-sitter-c`：AST 解析
- `apted`：树编辑距离
- `rapidfuzz`：模糊字符串匹配
- `scipy`：匈牙利算法

推荐评测维度：

#### 配置文件维度

* 顶层结构一致率
* 命令列表匹配率
* 帧处理器映射一致率
* 字段覆盖率

#### C 代码维度

* 函数结构相似度
* switch/case 分支匹配率
* 字段解析一致率
* 调用链相似度

#### 语义逻辑维度

* 处理的命令号/帧号是否一致
* 字段偏移量和类型是否一致
* 是否执行校验和封装逻辑
* 是否调用相同输出或分发流程

综合评分建议：

```text
final_score = 0.25 * config_score
            + 0.25 * code_structure_score
            + 0.50 * semantic_logic_score
```

---

## 6. 数据流设计

```text
源协议 gdwx 文档与代码
        │
        ▼
 文档解析 / 代码切块
        │
        ▼
   本地知识库构建
        │
        ▼
目标协议 dllg 文档输入
        │
        ▼
  协议结构化抽取
        │
        ├──────────────┐
        ▼              │
  检索源协议上下文        │
        │              │
        ▼              │
  生成 dllg_config.json │
        │              │
        └──────────► 生成 dllg_handler.c
                         │
                         ▼
                 与真实代码对比评测
                         │
                         ▼
                     报告输出
```

---

## 7. 配置设计

建议通过环境变量和配置文件统一管理：

### 7.1 环境变量

建议支持：

* `HERTA_API_BASE`
* `HERTA_API_KEY`
* `HERTA_MODEL`
* `HERTA_EMBED_MODEL`
* `HERTA_LOG_LEVEL`

### 7.2 配置文件

建议增加：

* `config/herta.yaml`

内容可包括：

* 默认源协议路径
* 默认目标协议路径
* 默认输出路径
* 检索 top-k
* embedding 模型
* 生成 temperature
* 是否保存中间 Prompt

---

## 8. Prompt 设计规范

Prompt 文件放在 `prompts/` 下，建议拆分为：

* `extract_protocol.md`
* `generate_config.md`
* `generate_handler.md`
* `compare_logic.md`

设计原则：

* Prompt 模板与代码分离。
* 每个 Prompt 只负责一个明确任务。
* 明确输出格式，不要混合说明和代码。
* 尽量要求模型输出可解析 JSON 或纯 C 代码。

---

## 9. 日志与中间产物

系统必须保留关键中间产物，便于复现和调试：

* 文档抽取文本
* 结构化协议 JSON
* 检索结果上下文
* Prompt 文本
* 原始模型响应
* 最终生成文件
* 对比评分 JSON
* Markdown 报告

日志建议分级：

* `INFO`：主流程状态
* `DEBUG`：检索结果、Prompt 长度、分块信息
* `ERROR`：解析失败、模型响应格式错误、文件读写失败

---

## 10. 错误处理原则

系统应对以下异常做清晰提示：

* 协议文档不存在
* 文档格式不支持
* 知识库未构建
* API Key 缺失
* 模型响应无法解析
* 对比输入文件缺失

CLI 应返回非零退出码，并输出可理解的错误信息。

---

## 11. 推荐实现优先级

### 第一阶段：最小可行版本

1. 实现 `buildkb`
2. 实现 `generate`
3. 实现 `compare`
4. 打通 `runall`

### 第二阶段：增强版本

1. 引入更强文档结构化抽取
2. 引入 AST 级 C 代码比较
3. 加入更细的相似度指标
4. 增加协议对照实验能力

---

## 12. 开发约束与实现原则

1. 所有模块尽量无状态。
2. 文件路径通过统一路径工具处理，不在业务代码中硬编码。
3. 生成与对比模块要可单独运行。
4. 先保证实验跑通，再逐步增强质量。
5. 所有对外接口优先返回结构化结果，而不是直接打印。

---

## 13. 验收标准

当系统满足以下条件时，视为架构目标达成：

1. 运行 `herta buildkb` 能成功构建 `gdwx` 知识库。
2. 运行 `herta generate` 能生成 `dllg_config.json` 和 `dllg_handler.c`。
3. 运行 `herta compare` 能输出对比报告。
4. 运行 `herta runall` 能一键完成完整实验。
5. 所有中间产物和报告都保存到预期目录。

---

## 14. 实现状态汇总

### 已完成功能

| 阶段 | 功能 | 状态 | 说明 |
|------|------|------|------|
| 1 | CLI 框架 | 完成 | Typer 框架，支持根命令和子命令 |
| 1 | 配置系统 | 完成 | 环境变量配置，支持智谱 AI API |
| 1 | 日志系统 | 完成 | 结构化日志输出 |
| 1 | 文档解析 | 完成 | 支持 docx/pdf 解析 |
| 1 | 代码解析 | 完成 | 支持 json/c/h 解析 |
| 1 | buildkb 命令 | 完成 | 支持四种分块策略 |
| 1 | 知识库构建 | 完成 | 分块 -> Embedding -> FAISS 索引 |
| 2 | generate config | 完成 | 单阶段生成 config.json |
| 2 | generate handler | 完成 | 两阶段生成 handler.c (plan + code) |
| 2 | 知识库检索 | 完成 | 向量相似度检索 |
| 3 | 四种分块策略 | 完成 | simple, semantic, auto-size, proposition |
| 4 | LLM 客户端 | 完成 | 支持 chat 和 embeddings，自动分批 |
| 5 | compare 子命令 | 完成 | 结果对比评测 CLI 命令 |
| 5 | compare/ 模块 | 完成 | JSON/C 代码对比实现，输出 metrics.json 和 comparison_report.md |
| 5 | JSON 比较 | 完成 | 拓扑/命令表/遥测表/内部处理 4 维度相似度 |
| 5 | C 代码 AST 比较 | 完成 | 基于 tree-sitter + 匈牙利算法函数匹配 |

### 预留/未实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| runall 子命令 | 预留 | 一键执行全流程 CLI 命令 |
| inspect 子命令 | 预留 | 查看知识库内容 CLI 命令 |
| utils/hash.py | 预留 | 哈希工具函数 |
| extract_protocol.md | 预留 | 协议抽取 Prompt 模板 |
| compare_logic.md | 预留 | 对比评测 Prompt 模板（基于 LLM 的语义对比） |

### 推荐后续优先级

1. **高优先级**：实现 runall 命令，支持一键执行完整实验
2. **中优先级**：优化 compare 的相似度算法，提升准确性
3. **低优先级**：实现 inspect 命令，用于查看知识库内容
4. **可选**：补充缺失的 Prompt 模板和工具函数

---

