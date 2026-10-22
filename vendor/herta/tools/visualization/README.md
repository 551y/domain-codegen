# Herta 知识库可视化工具集

本工具集提供知识库向量数据的人工可视化分析能力，**独立于 Herta 主流程运行**，需人工手动触发执行。

---

## 定位与设计理念

### 核心原则

1. **人工主导**：可视化不是自动化流程的一部分，必须由人工决定何时执行
2. **非侵入式**：不修改 Herta 主业务流程（buildkb/generate/runall 等）
3. **离线可用**：本地可视化功能优先保证离线可用，不强依赖联网服务
4. **基于现有产物**：输入完全来自已生成的知识库产物，不做重复计算

### 与主流程的关系

```
┌─────────────────────────────────────────────────────────────┐
│                     Herta 主流程                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────────┐  │
│  │ buildkb │ → │ generate│ → │ 其他步骤 │ → │ runall (完成) │  │
│  └─────────┘   └─────────┘   └─────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓ (知识库产物已生成)
┌─────────────────────────────────────────────────────────────┐
│                 人工可视化工具集（本目录）                    │
│                                                             │
│  ┌─────────────────┐     ┌─────────────────┐                 │
│  │ export_plotly.py│     │ export_atlas.py │                 │
│  │ 本地可视化       │     │ Atlas 数据导出   │                 │
│  │ （推荐优先）     │     │ （可选）         │                 │
│  └─────────────────┘     └─────────────────┘                 │
│                                                             │
│  ⚠️ 需人工手动执行，不自动触发                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 输入文件说明

工具集读取 `data/kb/<kb_name>/` 目录下的知识库产物：

| 文件 | 用途 | 必需 |
|------|------|------|
| `chunks.jsonl` | 文本块内容 + 元数据 | ✅ 必需 |
| `embeddings.npy` | 向量数据（用于降维） | ✅ 必需 |
| `meta.json` | 知识库元信息（辅助读取） | ⚠️ 可选 |
| `vector.index` | FAISS 索引（检索用） | ❌ 可视化不需要 |

### 为什么不需要 `vector.index`

- `vector.index` 是 FAISS 索引文件，用于高效向量检索
- 可视化需要的是原始 embedding 向量，不是索引结构
- Atlas 等可视化平台直接消费 embedding + metadata，不读取 FAISS 格式

---

## 安装依赖

```bash
# 进入工具集目录
cd tools/visualization

# 安装基础依赖
pip install -r requirements.txt
```

### 可选依赖

```bash
# 如需使用 UMAP 降维（推荐，效果更好）
pip install umap-learn

# 如需上传到 Nomic Atlas
pip install nomic
```

---

## 使用方式

### 1. 生成本地 Plotly 可视化（推荐）

```bash
python tools/visualization/export_plotly.py --kb gdwx
```

常用参数：

```bash
# 使用 PCA 降维（UMAP 不可用时自动回退）
python tools/visualization/export_plotly.py --kb gdwx --method pca

# 按类型（kind）着色
python tools/visualization/export_plotly.py --kb gdwx --color-by kind

# 手动指定输入输出路径
python tools/visualization/export_plotly.py \
    --kb gdwx \
    --kb-dir /path/to/kb \
    --out-dir /path/to/output
```

生成文件：

```
output/visualization/gdwx/
├── projection_2d.json       # 2D 投影坐标数据
├── plotly_scatter.html      # 交互式散点图（可直接打开）
└── visualization_meta.json    # 导出元信息
```

查看结果：

```bash
# Linux
open output/visualization/gdwx/plotly_scatter.html
```

### 2. 导出 Atlas 数据

```bash
python tools/visualization/export_atlas.py --kb gdwx
```

可选上传：

```bash
# 导出后尝试上传到 Atlas（需先安装 nomic 并登录）
python tools/visualization/export_atlas.py --kb gdwx --with-upload
```

生成文件：

```
output/visualization/gdwx/
├── atlas_payload.jsonl      # 文本数据（id, text, metadata）
├── atlas_metadata.csv       # 元数据表格
├── atlas_embeddings.npy     # embedding 向量（可选导出）
└── visualization_meta.json  # 导出元信息（合并更新）
```

Atlas 上传前置条件：

```bash
# 1. 安装 nomic 包
pip install nomic

# 2. 登录获取 API key
nomic login
# 按提示打开浏览器完成授权

```

---

## 常见问题

### Q1: chunks 和 embeddings 数量不一致怎么办？

**现象**：导出时报错 "chunks (100) 和 embeddings (95) 数量不一致"

**处理**：工具会自动截取较小的一方，使用前 N 条数据继续处理，并打印警告。

如需排查问题，可检查：

```bash
# 统计 chunks.jsonl 行数
wc -l data/kb/gdwx/chunks.jsonl

# 查看 embeddings 形状
python -c "import numpy as np; print(np.load('data/kb/gdwx/embeddings.npy').shape)"
```

### Q2: chunks.jsonl 缺少 section / kind 字段怎么办？

**处理**：工具会自动兜底填充：

- 尝试从 `metadata.section` / `metadata.kind` 读取
- 如仍缺失，填充为 `"unknown"`
- 可视化时按 `"unknown"` 类别着色

如需补充字段，可编辑原始数据或使用 `--color-by` 选择其他字段。

### Q3: UMAP 不可用如何回退？

**现象**：报错 "No module named 'umap'"

**处理**：

1. **自动回退**：如未安装 umap-learn，脚本会自动使用 PCA
2. **手动指定**：`--method pca`
3. **安装 UMAP**：`pip install umap-learn`

UMAP vs PCA：

- UMAP：保留局部结构更好，适合高维 embedding 可视化
- PCA：计算更快，不依赖额外包，效果尚可

### Q4: 着色类别太多导致图表混乱？

**处理**：工具会自动限制类别数量为 20 个，高频类别单独显示，其余归入"其他"。

如需调整：

- 修改 `export_plotly.py` 中的 `max_categories` 参数
- 或选择分布更均匀的字段（如 `kind` 通常比 `file` 类别少）

---

## 输出文件详细说明

### `projection_2d.json`

```json
{
  "coordinates": [[x1, y1], [x2, y2], ...],
  "record_ids": ["chunk_000001", "chunk_000002", ...],
  "method": "umap"
}
```

### `plotly_scatter.html`

- 单文件 HTML，可离线打开
- 交互功能：
  - 鼠标悬停查看 chunk 详情
  - 图例点击筛选类别
  - 缩放、平移
  - 保存为图片

### `atlas_payload.jsonl`

每行一条 JSON 记录：

```json
{
  "id": "gdwx_000001",
  "text": "原始文本内容...",
  "text_preview": "预览文本...",
  "file": "/path/to/source.docx",
  "section": "paragraph_0001",
  "kind": "doc",
  "chunk_strategy": "simple",
  "chunk_index": 0
}
```

### `atlas_metadata.csv`

CSV 表格，便于人工查看和过滤：

| id | file | section | kind | chunk_strategy | chunk_index | text_preview |
|----|------|---------|------|----------------|-------------|--------------|
| gdwx_000001 | /path/to/source.docx | paragraph_0001 | doc | simple | 0 | 预览文本... |

### `visualization_meta.json`

记录导出操作的元信息：

```json
{
  "kb_name": "gdwx",
  "kb_dir": "/path/to/data/kb/gdwx",
  "record_count": 302,
  "embedding_dim": 1024,
  "projection_method": "umap",
  "color_by": "file",
  "generated_files": [...],
  "generated_at": "2026-04-22T10:30:00"
}
```

---

## 技术说明

### 降维方法

- **UMAP**（默认）：非线性降维，更好地保留高维空间中的局部结构
  - 参数：min_dist=0.1, n_neighbors=15, metric="cosine"
  - 固定随机种子保证可复现
- **PCA**（回退）：线性降维，计算简单快速

### 兼容性

- Python 3.10+
- 不依赖 Herta 主代码（仅复用路径工具）
- 不依赖特定操作系统

---

## 注意事项

1. **不要将此工具集成到自动化流水线**
   - 不要在 `buildkb` 后自动执行
   - 不要在 `runall` 中调用
   - 保持人工决策何时可视化

2. **Atlas 上传是可选的**
   - 默认只导出数据，不强制联网
   - 没有 API key 时只生成本地文件
   - 离线环境可完全正常工作

3. **输出目录可安全删除**
   - 输出目录仅为可视化产物，不影响知识库本体
   - 可随时删除重新生成

---

## 文件清单

```
tools/visualization/
├── README.md           # 本文档
├── requirements.txt    # 依赖清单
├── common.py           # 公共数据加载与校验
├── export_plotly.py    # Plotly 可视化导出
└── export_atlas.py     # Atlas 数据导出
```
