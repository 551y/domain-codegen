# Herta 相似度计算方法详解

本文档详细介绍 `herta compare` 命令中 config.json 和 handler.c 的相似度计算方法，以及对应的代码实现位置。

---

## 目录

1. [概述](#概述)
2. [Config.json 相似度计算](#configjson-相似度计算)
3. [Handler.c 相似度计算](#handlerc-相似度计算)
4. [代码实现位置](#代码实现位置)
5. [相似度评分标准](#相似度评分标准)

---

## 概述

`herta compare` 命令对比 LLM 生成的结果与真实代码的相似度，输出两个核心指标：

- **config_similarity**: JSON 配置文件的相似度（0-100）
- **handler_similarity**: C 代码的相似度（0-100）
- **overall_similarity**: 整体相似度 = 0.35 × config + 0.65 × handler

---

## Config.json 相似度计算

### 计算方法概述

Config 相似度通过 **4 个维度** 的加权平均计算：

| 维度 | 权重 | 说明 |
|------|------|------|
| topology_similarity | 20% | JSON 树结构完整性 |
| cmd_list_similarity | 30% | 命令表匹配度 |
| tele_handler_similarity | 30% | 遥测表匹配度 |
| internal_handler_similarity | 20% | 内部处理表匹配度 |

**公式**:
```
config_similarity = 0.2×topology + 0.3×cmd_list + 0.3×tele_handler + 0.2×internal
```

### 1. Topology Similarity（拓扑相似度）

**目的**: 检查 JSON 的 4 层树结构是否完整

**检查层级**:
- **第 1 层**: 根节点必须为 object/dict
- **第 2 层**: 必须包含 6 个字段
  - `version`, `author`, `start_index`
  - `cmd_list`, `tele_handler`, `internal_handler`
- **第 3 层**: 三个列表字段必须为 list 且非空
- **第 4 层**: 列表元素必须包含指定字段
  - `cmd_list` 元素: `desc`, `cmd`, `handler`
  - `tele_handler` 元素: `desc`, `len`, `handler`
  - `internal_handler` 元素: `desc`, `len`, `handler`

**评分**: 每项检查通过得分，满分 100 分

### 2. Cmd List Similarity（命令表相似度）

**对齐方式**: 按 `cmd` 字段对齐命令项

**比较项**:
- `cmd` 是否存在
- `handler` 是否**精确匹配**
- `desc` 是否**模糊匹配**（使用 rapidfuzz 计算相似度）

**单项得分计算**:
```
score = 40×handler_match + 40×(desc_similarity/100) + 20×cmd_exists
```

### 3. Tele Handler Similarity（遥测表相似度）

**对齐方式**: 按 `handler` 字段对齐遥测项

**比较项**:
- `handler` 是否一致
- `len` 是否一致
- `desc` 是否一致或近似

### 4. Internal Handler Similarity（内部处理表相似度）

**对齐方式**: 按 `handler` 字段对齐内部处理项

**比较项**:
- `handler` 是否一致
- `len` 是否一致
- `desc` 是否一致或近似

---

## Handler.c 相似度计算

### 计算方法概述

Handler 相似度通过 **AST 级函数匹配** 计算：

1. **提取函数**: 使用 tree-sitter 解析 C 代码，提取所有函数
2. **计算相似度**: 对每对函数计算 AST 相似度
3. **函数匹配**: 使用匈牙利算法求解最优一对一匹配
4. **综合评分**: 基于匹配结果计算整体相似度

### 1. AST 解析

**工具**: tree-sitter (C grammar)

**提取信息**:
- 函数名
- 返回类型
- 参数列表
- 函数体源码
- 函数体 AST

### 2. 函数相似度计算

**公式**:
```
ast_similarity = 0.70 × tree_edit_similarity + 0.30 × semantic_feature_similarity
```

#### Tree Edit Similarity（树编辑相似度）

**算法**: APTED (Adaptive Tree Edit Distance)

**步骤**:
1. 将函数 AST 转换为轻量树结构
2. 使用 APTED 计算树编辑距离
3. 归一化为 0-1 相似度:
   ```
   similarity = 1 - (distance / (max_nodes × 2))
   ```

**备选方案**: 如果 apted 不可用，使用简化计算（节点数量和类型统计）

#### Semantic Feature Similarity（语义特征相似度）

**提取的特征**:
1. **调用函数集合**: 函数体内调用的其他函数
2. **字面量常量集合**: 数字、字符串常量
3. **赋值左值模式**: 被赋值的变量模式
4. **标识符集合**: 所有标识符
5. **Return 语句模式**: return 的类型分类
6. **控制流模式**: if/else/for/while/switch 的数量

**计算方法**: Jaccard 相似度（集合交集/并集）

### 3. 函数匹配

**算法**: 匈牙利算法 (Hungarian Algorithm)

**实现**: `scipy.optimize.linear_sum_assignment`

**流程**:
1. 构建相似度矩阵: 每个真实函数 vs 每个生成函数
2. 转换为代价矩阵: `cost = 1 - similarity`
3. 求解最优匹配: 使整体相似度最大
4. 应用阈值: 相似度 < 0.3 视为未匹配

**匹配约束**: 一对一匹配，避免一个生成函数被多个真实函数占用

### 4. Handler 整体相似度

**计算公式**:
```
handler_similarity = (avg_similarity × 0.80 + coverage × 0.10 + name_match_rate × 0.10) × 100
```

**组成**:
- **avg_similarity**: 所有匹配函数的平均 AST 相似度 (80%)
- **coverage**: 匹配函数数 / 真实函数总数 (10%)
- **name_match_rate**: 同名匹配数 / 匹配函数数 (10%)

---

## 代码实现位置

### Compare 模块目录结构

```
src/herta/compare/
├── __init__.py              # 模块导出
├── json_compare.py          # JSON 配置对比
├── c_ast_parser.py          # C 代码 AST 解析
├── ast_similarity.py      # AST 相似度计算
├── function_matcher.py      # 函数匹配算法
└── report_builder.py        # 报告生成
```

### 各功能代码位置

#### 1. Config.json 相似度计算

**文件**: `src/herta/compare/json_compare.py`

**核心类**: `ConfigComparator`

**关键方法**:
- `_compare_topology()`: 拓扑相似度计算（约 80 行）
- `_compare_cmd_list()`: 命令表相似度计算（约 60 行）
- `_compare_tele_handler()`: 遥测表相似度计算（约 50 行）
- `_compare_internal_handler()`: 内部处理表相似度计算（约 50 行）

**数据结构**: `ConfigComparisonResult` - 存储 4 个维度的比较结果

**使用的第三方库**:
- `rapidfuzz`: 模糊字符串匹配（desc 字段比较）

#### 2. Handler.c AST 解析

**文件**: `src/herta/compare/c_ast_parser.py`

**核心类**: `CASTParser`

**关键方法**:
- `parse_file()`: 解析 C 文件，提取所有函数（约 30 行）
- `_extract_function()`: 从 AST 节点提取函数信息（约 40 行）
- `_extract_function_name()`: 提取函数名（约 30 行）
- `_extract_return_type()`: 提取返回类型（约 20 行）
- `_extract_parameters()`: 提取参数列表（约 30 行）

**数据结构**: `FunctionInfo` - 存储函数名、返回类型、参数、函数体源码、AST

**使用的第三方库**:
- `tree-sitter`: AST 解析框架
- `tree-sitter-c`: C 语言 grammar

#### 3. AST 相似度计算

**文件**: `src/herta/compare/ast_similarity.py`

**核心类**: `ASTSimilarityCalculator`

**关键方法**:
- `calculate()`: 主入口，计算两个函数 AST 的相似度（约 30 行）
- `_calculate_tree_edit_similarity()`: 树编辑距离相似度（约 50 行）
  - 使用 `apted` 库
  - 备选：简化树相似度计算
- `_calculate_semantic_similarity()`: 语义特征相似度（约 80 行）
  - `SemanticFeatureExtractor` 类提取 6 种特征
  - Jaccard 相似度计算

**数据结构**: `SimilarityResult` - 存储 tree_edit、semantic、ast 相似度

**使用的第三方库**:
- `apted`: 树编辑距离算法
- `collections.Counter`: 特征统计

#### 4. 函数匹配

**文件**: `src/herta/compare/function_matcher.py`

**核心类**: `FunctionMatcher`

**关键方法**:
- `match()`: 主入口，执行函数匹配（约 60 行）
- `_build_similarity_matrix()`: 构建相似度矩阵（约 20 行）
- `_optimal_matching()`: 最优匹配（约 80 行）
  - 使用 `scipy.optimize.linear_sum_assignment`
  - 备选：贪心匹配算法
- `_calculate_handler_similarity()`: 计算整体 handler 相似度（约 30 行）

**数据结构**: 
- `FunctionMatch` - 单个函数匹配结果
- `FunctionMatchingResult` - 完整匹配结果

**使用的第三方库**:
- `scipy.optimize.linear_sum_assignment`: 匈牙利算法
- `numpy`: 矩阵操作

**匹配参数**:
- `SIMILARITY_THRESHOLD = 0.3`: 相似度阈值

#### 5. 报告生成

**文件**: `src/herta/compare/report_builder.py`

**核心类**: `ReportBuilder`

**关键方法**:
- `generate_metrics()`: 生成 metrics.json 数据结构（约 30 行）
- `generate_markdown_report()`: 生成 Markdown 报告（约 150 行）
- `save_reports()`: 保存报告到文件（约 20 行）

**输出文件**:
- `metrics.json`: 结构化指标
- `comparison_report.md`: 可读性报告

#### 6. CLI 命令

**文件**: `src/herta/cli/compare.py`

**功能**: 实现 `herta compare` 命令

**关键逻辑**:
1. 解析命令行参数
2. 推断默认路径
3. 调用 `json_compare.compare_configs()`
4. 调用 `c_ast_parser.parse_c_functions()`
5. 调用 `function_matcher.match_functions()`
6. 调用 `report_builder.generate_comparison_report()`
7. 输出结果摘要

### 依赖列表

```txt
# 新增依赖（用于 compare 模块）
tree-sitter>=0.21.0          # AST 解析框架
tree-sitter-c>=0.21.0        # C 语言 grammar
apted>=1.0.3                 # 树编辑距离算法
rapidfuzz>=3.0.0             # 模糊字符串匹配
scipy>=1.12.0                # 匈牙利算法
```

---

## 相似度评分标准

| 分数范围 | 评价 | 建议 |
|---------|------|------|
| 80-100 | 优秀 | 生成结果与真实代码高度相似，可直接使用或微调 |
| 60-79 | 良好 | 生成结果基本可用，部分细节需要人工调整 |
| 40-59 | 一般 | 生成结果需要较多修改，建议检查生成参数 |
| 0-39 | 较差 | 生成结果质量低，建议重新生成或调整策略 |

---

## 示例输出

### metrics.json 片段

```json
{
  "target": "dllg",
  "topology_similarity": 85.0,
  "cmd_list_similarity": 78.5,
  "tele_handler_similarity": 82.0,
  "internal_handler_similarity": 75.5,
  "config_similarity": 80.2,
  "handler_similarity": 72.8,
  "overall_similarity": 75.4,
  "function_match_results": [
    {
      "gt_func_name": "handle_query_status",
      "gen_func_name": "handle_query_status",
      "is_name_match": true,
      "ast_similarity": 0.85
    }
  ]
}
```

### comparison_report.md 结构

```markdown
# Herta 对比评测报告 - dllg

## 总体评分
整体相似度: 75.4%

## Config 详细分析
### 1. 拓扑结构相似度: 85.0%
...

## Handler 代码详细分析
### 函数匹配结果
| 真实函数 | 生成函数 | 同名 | AST 相似度 | 状态 |
...
```

---

## 扩展阅读

- [architecture.md](./architecture.md): 系统整体架构
- [cli.md](./cli.md): CLI 命令详细说明
- 代码文件:
  - `src/herta/compare/json_compare.py`
  - `src/herta/compare/c_ast_parser.py`
  - `src/herta/compare/ast_similarity.py`
  - `src/herta/compare/function_matcher.py`
  - `src/herta/compare/report_builder.py`
  - `src/herta/cli/compare.py`
