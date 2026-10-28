# 领域知识增强的无人系统代码智能适配平台 — 集成演示（domain-codegen）

> 一套 RAG 代码生成框架，两套领域知识库，按场景注入：
> - **入口一 · 卫星协议代码生成**：星务协议知识库（3090 知识块，FAISS）；协议文档 docx → 检索增强 → LLM 生成 config.json + handler.c → 自动评测（配置四维 + AST 结构相似度）
> - **入口二 · 无人机模型部署适配**：ONNX 知识库（1349 条，9 类）与 SemTrans PyTorch/Paddle 数学定义独立维护；PyTorch 源码 .py，进入后以「目标框架」开关二选一——
>   - **PyTorch→ONNX 部署导出**：BM25 离线检索 → LLM 生成导出代码 → **子进程真实执行导出，产出可下载的单文件 .onnx 模型**（含权重；torch 2.x 导出的外部权重自动重打包内嵌）→ **按模型真实 IO 元数据实例化机载侧 C++ 推理源码**（ONNX Runtime C++ API，动态读取输入输出，含编译/预处理说明注释）→ 评测（语法/导出调用/导出执行/onnx.checker 校验）
>   - **PyTorch→PaddlePaddle 跨框架转换**：SemTrans 经配置的 OpenAI-compatible API 生成 Paddle 候选 → PyTorch/Paddle frontend 与 SemTrans-JIT 数学定义执行区域语义对齐 → 基于诊断证据定向修复 → 使用独立输入复验。结果明确区分通过、失败与未验证

## 启动

```bash
./start.sh    # 后端 :8200 + 前端 :3200，访问 http://localhost:3200
./stop.sh
```

Windows（原生 PowerShell）：`.\start.ps1` / `.\stop.ps1`，安装步骤见下文「重新部署 · Windows」。

## 演示流程（工作台一页走完）

1. **选择知识库**：两张场景卡（卫星协议代码生成 / 无人机模型部署适配）；模型入口内以「目标框架」开关选择 PyTorch→ONNX 部署导出或 PyTorch→PaddlePaddle SemTrans 演示。
2. **输入**：协议和 ONNX 沿用现有样例/上传方式；SemTrans 展示工作目录已有的真实 AlexNet、DenseNet、ShuffleNetV2 PyTorch 模型，并由对应 profile 提供构造参数和 dummy 输入形状。上传 Paddle 源码时填写主类、dummy 输入形状和构造参数。
3. **开始生成**：四阶段进度（解析输入 → 检索知识库 → 生成目标代码 → 评测结果），**执行过程中实时展示中间输出**——解析到的算子调用、检索命中的知识条目（标题/类别/相关度）、协议 config 阶段产物等
4. **结果与评测**：ONNX 保持既有模型与导出结果；SemTrans 展示初始候选、最终 Paddle 源码、区域对齐诊断、修复轮次、独立输入复验、容差、样本数及耗时。SemTrans 检查是有限输入的执行证据，不是形式化证明。

**多任务并行**：在线生成（LLM 真实调用，数分钟）运行期间，工作台保持可操作——可同时运行/浏览离线预置样例；所有任务以「任务条」标签呈现进度，点击随时切换查看，同一输入重复提交会被去重。

「知识库」页可分别浏览星务协议知识块（3090）、ONNX 模型部署适配知识（1349，9 类）与 SemTrans PyTorch/Paddle 数学定义目录。SemTrans 定义目录基于包内原始 YAML 生成，不是模型准确率数据集。

## LLM 配置

- 顶栏右侧齿轮是唯一模型服务设置入口，服务以图标卡片选择：ZCode（智谱）、Codex（OpenAI API）和 DeepSeek。三者的 Base URL、模型和 API Key 独立保存；协议、PyTorch→ONNX 和 PyTorch→Paddle（SemTrans）使用当前选中的服务。
- API Key 由后端保存到工作区外的 `%LOCALAPPDATA%\\DomainCodegen\\model-provider-settings.json`，使用当前 Windows 用户 DPAPI 加密；接口只返回每个服务是否已配置，页面重开后不会回显任何已保存的 Key。切换服务不会覆盖其他服务的密钥。
- 协议/ONNX 的预置样例走预生成回放，上传文件才走真实 LLM 生成（需配置 Key）。
- SemTrans 初始候选与定向修复实际调用当前服务对应的官方 OpenAI API/兼容接口（Codex 使用 Responses，ZCode/DeepSeek 使用 Chat Completions）；连接失败、生成失败、语义差异、未找到有效输入及超时都有独立状态，不回退成 ModelX 或预生成 Paddle 结果。设置页区分普通标准 API Key、Coding Plan Key 和其他服务 Key；Coding Plan Key 会被阻止发送到本平台，因为智谱官方限定其仅用于指定工具/产品环境。
- SemTrans 真实视觉模型使用紧凑候选提示，避免长篇 Chain-of-Thought 让大模型在源码转换前超时；若候选生成成功但 Linear/BatchNorm 参数布局不一致，页面会显示语义失败及具体参数形状证据，不会把“代码生成成功”误判成“转换通过”。
- **检索不依赖 Key**：ONNX 知识库的检索为后端本地 BM25 + bigram 混合（离线真实执行，演示时进度条走到「检索知识库」即真实检索）
- SemTrans 语义验证由本地 PyTorch/Paddle frontend、SemTrans-JIT 和动态区域对齐执行；候选生成/修复由配置的 OpenAI-compatible 模型服务完成。只有完成有效输入和独立复验的语义检查才显示通过；模型公式缺失或预算内无有效样本显示未验证。此评测不等同论文全量实验或形式化等价证明。

## 生成速度优化（在线生成）

- **并行生成**：协议路线的 config 与 handler 两条生成链相互独立，已并行执行（wall-clock 从两者之和降为两者之较长者）
- **检索注入而非全量注入**：协议知识库已从 direct（201 块全量注入每次调用）切换为 semantic（FAISS 向量检索 top_k 注入），提示词缩小约 85%，单次 LLM 调用显著提速；想回到全量注入把 `vendor/herta/data/kb/gdwx/meta.json` 的 `chunk_strategy` 改回 `"direct"` 即可
- **模型选择**：顶栏齿轮可换用更小更快的模型（如 air/flash 档），是剩余最大的速度杠杆；生成质量与速度按演示需要取舍

## 数据与口径

- ONNX 知识库来源：`vendor/onnx/dataset.expanded.jsonl`（pytorch-onnx-dataset2 交付版，1349 条：算子映射 205 / 复合算子实现 347 / 测试用例 327 / 基准测试 219 / 版本兼容矩阵 80 / 转换示例 50 / 技术文档 49 / 错误模式 40 / 修复案例 32；另有 `manifest.json` 记录口径与指纹）
- 星务协议知识库来源：`协议代码数据集2/dataset/augmented`（3090 条增强版：gdwx 253 + dllg 2837，细分块 150 字符/重叠 30，向量随数据集交付），已转换为 herta KB 构建产物格式内置于 `vendor/herta/data/kb/industrial_proto/`；预生成结果与 herta 引擎同目录；PyTorch 样例源码内置于 `vendor/opalign/`（原始出处分别为 `../demo/herta`、`../demo/opalign/EVALUATION`）
- 预生成 ONNX 导出脚本：`server/preset_onnx/*_to_onnx.py`（12 个视觉样例；此路线本次未改变）。
- SemTrans 演示 profile 位于 `server/semtrans_demo_cases.json`，直接引用工作区已有 `vendor/opalign/EVALUATION/datasets/models/pytorch/vision/sourceModels` 中的真实 AlexNet、DenseNet、ShuffleNetV2 源码及构造参数；不再把自造 Tiny 模型作为前端样例。受控 AvgPool 差异仅保留在 `server/tests/fixtures/` 的回归测试中，不作为产品转换样例。
- SemTrans 源码闭包位于 `vendor/semtrans/SemTrans/`，来自版本 `431bd23c9773ec4a3c2d9d924e433e93d5a5577b`。PyTorch/Paddle 的 `nn_ops.yaml` 和 `tensors.yaml` 原样复制；`vendor/semtrans/provenance.json` 记录逐文件来源哈希及当前 6 项本地适配说明。当前原项目 README 声明 SemTrans 本体尚无公开许可证，因此本仓库目前仅用于本地集成与审核，未经权利人许可不要向公开远端发布这些原代码。
- `server/tools/build_semtrans_dataset.py` 从两侧原始定义文件生成独立的语义定义视图：实际条目由重复定义合并/同名匹配计算，不代表算子准确率或支持率。源文件更新后运行该脚本重建。
- SemTrans 依赖见 `vendor/semtrans/requirements.txt`。已验证的 worker 环境为 Windows / Python 3.9 / PyTorch 2.7.0+cu126（CPU 执行）/ Paddle 2.6.1 / SciPy 1.13.1 / SymPy 1.x / Z3 4.15.4。其余 Python/Paddle 组合须自行验证。若后端运行环境不同，可通过环境变量 `SEMTRANS_PYTHON` 指定 worker Python 可执行文件；此变量只指定解释器，不承载密钥。
- SemTrans 核心真实回归：`python server/tools/verify_semtrans_alignment.py`；验证 PyTorch/Paddle 小网络通过及 AvgPool 数值差异被真实 `auto_repair_pipeline` / frontend / SemanticTracer 发现。数值比较对每个输出检查结构、shape、dtype、有限值、误差与容差；通过报告包含实际区域对齐和最大误差。该回归不代表模型服务在线端到端生成已验证。
- 三个服务的官方默认地址为：ZCode `https://open.bigmodel.cn/api/paas/v4` / `glm-5.3`，Codex `https://api.openai.com/v1` / `gpt-5-codex`（Responses API），DeepSeek `https://api.deepseek.com` / `deepseek-chat`（Chat Completions）；模型 ID 可按账号实际可用模型调整。连接测试按各服务对应的官方协议发送；Coding Plan Key 仍会被拒绝发送。参考 [ZCode 文档](https://docs.bigmodel.cn/cn/coding-plan/tool/zcode)、[OpenAI API](https://platform.openai.com/docs/api-reference/responses) 与 [DeepSeek API](https://api-docs.deepseek.com/)。
- ZCode 的 `glm-5.3` 会先生成 reasoning；连接探针保留 512 token 预算，避免过小的探针上限耗尽后把空正文误判成 Key/端点故障。Paddle 单次生成/修复超时默认为 150 秒、整任务默认为 360 秒；可分别通过 `SEMTRANS_LLM_TIMEOUT_S`、`SEMTRANS_TASK_TIMEOUT_S` 调整。真实生成仍可能受服务延迟影响。
- 协议评测分数为现场真实重算口径，与历史批次、年度报告综合口径略有差异

## 重新部署（换机器，需有网一次）

**整个平台已自包含：迁移只需拷贝 `domain-codegen/` 一个目录**（引擎、三套知识库数据、演示用例 `case/` 全部内置），对放置位置无要求。

### Linux / macOS / WSL

```bash
conda create -n demo python=3.12 -y && conda activate demo
pip install -e ./vendor/herta        # 引擎与依赖（fastapi/uvicorn/faiss/tree-sitter 等）
pip install python-multipart         # 上传接口需要
pip install torch onnx onnxscript onnxruntime   # ONNX 实导出 + 冒烟测试需要（CPU 版即可；onnxscript 是 torch.onnx 的硬依赖，缺省时回退为仅交付导出脚本）
pip install -r ./vendor/semtrans/requirements.txt
conda create -n semtrans-worker python=3.9 -y
conda run -n semtrans-worker pip install torch==2.7.0 paddlepaddle==2.6.1
conda run -n semtrans-worker pip install -r ./vendor/semtrans/requirements.txt
# 启动前将 SEMTRANS_PYTHON 设为该环境的 python 可执行文件
cd app && npm install --legacy-peer-deps && npm run build
```

### Windows（原生，无 WSL）

启动脚本为 `start.ps1` / `stop.ps1`（PowerShell）。先装好 [Node.js LTS](https://nodejs.org) 与 Miniconda，然后：

```powershell
conda create -n demo python=3.12 -y; conda activate demo
conda install -c conda-forge faiss-cpu -y   # faiss-cpu 没有 Windows 的 pip 包，须走 conda-forge（先装它，后续 pip 会识别为已满足）
pip install -e ./vendor/herta
pip install python-multipart
pip install torch onnx onnxscript onnxruntime   # ONNX 实导出 + 冒烟测试需要（onnxscript 为 torch.onnx 硬依赖）
pip install -r .\vendor\semtrans\requirements.txt
conda create -n semtrans-worker python=3.9 -y
conda run -n semtrans-worker pip install torch==2.7.0 paddlepaddle==2.6.1
conda run -n semtrans-worker pip install -r .\vendor\semtrans\requirements.txt
cd app; npm install --legacy-peer-deps; npm run build; cd ..
# 启动前指定 worker 解释器；路径按本机 Conda 安装位置调整
$env:SEMTRANS_PYTHON = "C:\path\to\anaconda3\envs\semtrans-worker\python.exe"
.\start.ps1     # 若提示禁止运行脚本：powershell -ExecutionPolicy Bypass -File .\start.ps1
.\stop.ps1
```

SemTrans 对 PyTorch→Paddle 运行所用 Python 解释器默认与 FastAPI 后端相同。上面的隔离 worker 环境是已验证的安装口径（Python 3.9 / PyTorch 2.7.0 / PaddlePaddle 2.6.1 / CPU 推理）；其余组合需实测。若框架依赖必须隔离，设置 `SEMTRANS_PYTHON` 为装有 PyTorch、Paddle 及 `vendor/semtrans/requirements.txt` 依赖的 worker Python 可执行文件后再启动服务。随后在页面右上角设置中选择 API Key 类别，填写相应官方 OpenAI-compatible Base URL、模型 ID 和 API Key，再执行连接测试。Coding Plan Key 不可用于本平台的自建网页调用。Windows DPAPI 密钥由运行后端的当前用户账户解密，换账户或换机器需要重新配置。

Windows 版日志为 `logs/server.log` + `logs/server.err.log`（前端同理，`Start-Process` 不支持 stdout/stderr 合并到同一文件，故分两个）。

## 故障处置

| 现象 | 处置 |
|---|---|
| 生成失败提示未配置 Key | 预期（离线），换预置样例或配置 Key |
| 端口占用 | `./stop.sh`（Windows：`.\stop.ps1`）后重起 |
| Windows 报「应用程序控制策略已阻止此文件」 | 脚本已改用 `python -m uvicorn` 启动规避（pip 的 uvicorn.exe 启动器无签名易被拦）；若 node 也被拦，换 Node.js 官网 LTS 安装包重装 |
| 页面空数据 | `curl localhost:8200/api/health` 确认后端 |
