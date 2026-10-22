# 领域知识增强的无人系统代码智能适配平台 — 集成演示（domain-codegen）

> 一套 RAG 代码生成框架，两套领域知识库，按场景注入：
> - **入口一 · 星务协议知识库**（201 知识块，FAISS）：协议文档 docx → 检索增强 → LLM 生成 config.json + handler.c → 自动评测（配置四维 + AST 结构相似度）
> - **入口二 · PyTorch→ONNX 知识库**（1814 条，9 类）：PyTorch 源码 .py → BM25 离线检索 ONNX 知识 → LLM 生成 ONNX 导出代码 → 自动评测（语法/导出调用/自检）

## 启动

```bash
./start.sh    # 后端 :8200 + 前端 :3200，访问 http://localhost:3200
./stop.sh
```

Windows（原生 PowerShell）：`.\start.ps1` / `.\stop.ps1`，安装步骤见下文「重新部署 · Windows」。

## 演示流程（工作台一页走完）

1. **选择知识库**：两张场景卡（卫星协议代码生成 / 无人机模型部署适配 PyTorch→ONNX）
2. **输入**：预置样例（带「完整演示」角标：协议 dllg；算子 alexnet / densenet / vgg）或拖拽上传 .docx/.py（`case/` 目录内置两个现成上传用例：星务协议 docx、DroneNet 模型源码 py，见 `case/README.md`）
3. **开始生成**：四阶段进度（解析输入 → 检索知识库 → 生成目标代码 → 评测结果），**执行过程中实时展示中间输出**——解析到的算子调用、检索命中的知识条目（标题/类别/相关度）、协议 config 阶段产物等
4. **结果与评测**：左侧生成代码，右侧自动评测面板（分数条 + 明细 + 检索注入的知识列表）

**多任务并行**：在线生成（LLM 真实调用，数分钟）运行期间，工作台保持可操作——可同时运行/浏览离线预置样例；所有任务以「任务条」标签呈现进度，点击随时切换查看，同一输入重复提交会被去重。

「知识库」页可分别浏览两套知识库：协议知识块（201）与 ONNX 知识记录（1814，9 类），均可搜索。

## LLM 配置

- 顶栏右侧齿轮：配置 API Key / Base URL / 模型（进程内生效，重启恢复环境变量）
- **路径规则（重要）**：**预置样例永远走预生成回放，无论是否配置 Key 都不调 LLM**（秒级出结果，回放的是真实流水线的历史生成产物）；**只有上传自有文件才走真实 LLM 生成**（需配置 Key，未配置会在生成阶段明确报错）
- **检索不依赖 Key**：ONNX 知识库的检索为后端本地 BM25 + bigram 混合（离线真实执行，演示时进度条走到「检索知识库」即真实检索）
- 评测**始终是本地真实计算**：协议路线为配置四维加权 + AST 函数匹配（树编辑距离计算量大，已按内容指纹缓存——同输入第二次秒出）；算子路线为语法检查 + 导出调用完备性 + 自检检查

## 数据与口径

- ONNX 知识库来源：`vendor/onnx/dataset.standardized.jsonl`（1814 条：算子映射 205 / 复合算子实现 420 / 测试用例 776 / 基准测试 243 / 技术文档 49 / 修复案例 32 / 版本矩阵 80 / 错误模式 8 / 示例 1）
- 协议知识库、预生成结果与 herta 引擎内置于 `vendor/herta/`；PyTorch 样例源码内置于 `vendor/opalign/`（原始出处分别为 `../demo/herta`、`../demo/opalign/EVALUATION`）
- 预生成导出脚本：`server/preset_onnx/{alexnet,densenet,vgg}_to_onnx.py`（densenet 脚本含 checkpoint 不可 trace 与 MaxPool dilation 不支持两个真实导出难点的处理注释）
- 算子样例输入为 PyTorch 模型源码（EVALUATION sourceModels，过滤 `__init__` 后 18 个）
- 协议评测分数为现场真实重算口径，与历史批次、年度报告综合口径略有差异

## 重新部署（换机器，需有网一次）

**整个平台已自包含：迁移只需拷贝 `domain-codegen/` 一个目录**（引擎、两套知识库数据、演示用例 `case/` 全部内置），对放置位置无要求。

### Linux / macOS / WSL

```bash
conda create -n demo python=3.12 -y && conda activate demo
pip install -e ./vendor/herta        # 引擎与依赖（fastapi/uvicorn/faiss/tree-sitter 等）
pip install python-multipart         # 上传接口需要
cd app && npm install --legacy-peer-deps && npm run build
```

### Windows（原生，无 WSL）

启动脚本为 `start.ps1` / `stop.ps1`（PowerShell）。先装好 [Node.js LTS](https://nodejs.org) 与 Miniconda，然后：

```powershell
conda create -n demo python=3.12 -y; conda activate demo
conda install -c conda-forge faiss-cpu -y   # faiss-cpu 没有 Windows 的 pip 包，须走 conda-forge（先装它，后续 pip 会识别为已满足）
pip install -e ./vendor/herta
pip install python-multipart
cd app; npm install --legacy-peer-deps; npm run build; cd ..
.\start.ps1     # 若提示禁止运行脚本：powershell -ExecutionPolicy Bypass -File .\start.ps1
.\stop.ps1
```

Windows 版日志为 `logs/server.log` + `logs/server.err.log`（前端同理，`Start-Process` 不支持 stdout/stderr 合并到同一文件，故分两个）。

## 故障处置

| 现象 | 处置 |
|---|---|
| 生成失败提示未配置 Key | 预期（离线），换预置样例或配置 Key |
| 端口占用 | `./stop.sh`（Windows：`.\stop.ps1`）后重起 |
| Windows 报「应用程序控制策略已阻止此文件」 | 脚本已改用 `python -m uvicorn` 启动规避（pip 的 uvicorn.exe 启动器无签名易被拦）；若 node 也被拦，换 Node.js 官网 LTS 安装包重装 |
| 页面空数据 | `curl localhost:8200/api/health` 确认后端 |
