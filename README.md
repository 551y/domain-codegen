# 领域知识增强的无人系统代码智能适配平台 — 集成演示（domain-codegen）

> 一套 RAG 代码生成框架，两套领域知识库，按场景注入：
> - **入口一 · 卫星协议代码生成**：星务协议知识库（3090 知识块，FAISS）；协议文档 docx → 检索增强 → LLM 生成 config.json + handler.c → 自动评测（配置四维 + AST 结构相似度）
> - **入口二 · 无人机模型部署适配**：ONNX 知识库（1349 条，9 类）与 ModelX Paddle 映射规则（541 条）独立维护；PyTorch 源码 .py，进入后以「目标框架」开关二选一——
>   - **PyTorch→ONNX 部署导出**：BM25 离线检索 → LLM 生成导出代码 → **子进程真实执行导出，产出可下载的单文件 .onnx 模型**（含权重；torch 2.x 导出的外部权重自动重打包内嵌）→ **按模型真实 IO 元数据实例化机载侧 C++ 推理源码**（ONNX Runtime C++ API，动态读取输入输出，含编译/预处理说明注释）→ 评测（语法/导出调用/导出执行/onnx.checker 校验）
>   - **PyTorch→PaddlePaddle 跨框架转换**：**ModelX CMC 源码级转换**（使用自身 AST/映射规则，不依赖 LLM）→ 语法/框架迁移彻底性/结构完整性评测、ModelX API 成功率统计与 Paddle dummy 前向冒烟；额外检索 ONNX 知识库仅作参考展示，不参与转换

## 启动

```bash
./start.sh    # 后端 :8200 + 前端 :3200，访问 http://localhost:3200
./stop.sh
```

Windows（原生 PowerShell）：`.\start.ps1` / `.\stop.ps1`，安装步骤见下文「重新部署 · Windows」。

## 演示流程（工作台一页走完）

1. **选择知识库**：两张场景卡（卫星协议代码生成 / 无人机模型部署适配）；模型入口内以「目标框架」开关选择 PyTorch→ONNX 部署导出或 PyTorch→PaddlePaddle 跨框架转换——样例/上传输入共用，转换规则与评测按目标分别处理
2. **输入**：预置样例（带「完整演示」角标：协议 dllg；算子 alexnet / densenet / vgg）或拖拽上传 .docx/.py（`case/` 目录内置两个现成上传用例：星务协议 docx、DroneNet 模型源码 py，见 `case/README.md`）
3. **开始生成**：四阶段进度（解析输入 → 检索知识库 → 生成目标代码 → 评测结果），**执行过程中实时展示中间输出**——解析到的算子调用、检索命中的知识条目（标题/类别/相关度）、协议 config 阶段产物等
4. **结果与评测**：左侧生成代码，右侧模型产物卡 + 自动评测面板——ONNX 目标产出可下载的 .onnx 模型文件（大小 / opset / 输入输出形状 / checker 状态一览）与配套 C++ 推理源码页签，Paddle 目标产出转换后模型源码；评测含分数条 + 明细 + 检索注入的知识列表

**多任务并行**：在线生成（LLM 真实调用，数分钟）运行期间，工作台保持可操作——可同时运行/浏览离线预置样例；所有任务以「任务条」标签呈现进度，点击随时切换查看，同一输入重复提交会被去重。

「知识库」页可分别浏览星务协议知识块（3090）、ONNX 模型部署适配知识（1349，9 类）与 ModelX Paddle 转换规则（541），均可搜索。规则条目不是带标签的性能/精度测试数据集。

## LLM 配置

- 顶栏右侧齿轮：配置 API Key / Base URL / 模型（进程内生效，重启恢复环境变量）
- 协议/ONNX 的预置样例走预生成回放，上传文件才走真实 LLM 生成（需配置 Key）。
- **路径规则（重要）**：Paddle 样例和上传文件都优先走内置 ModelX CMC（确定性转换，不调用 LLM）；样例在 ModelX 依赖缺失时回退到随仓库交付的预生成产物，上传文件则明确提示安装依赖
- **检索不依赖 Key**：ONNX 知识库的检索为后端本地 BM25 + bigram 混合（离线真实执行，演示时进度条走到「检索知识库」即真实检索）
- 评测**始终是本地真实计算**：协议路线预置样例为配置四维加权 + AST 函数匹配（树编辑距离计算量大，已按内容指纹缓存——同输入第二次秒出）；**上传协议文档无人工标准答案，改为「参照文档的覆盖度评测」**——从上传文档表格中提取命令码/遥测信号/处理函数名，与生成产物做名称级匹配（F1），外加字段完整性与 Handler 语法检查；ONNX 路线为语法 + 导出执行 + checker/冒烟检查；Paddle 路线为 ModelX API 转换统计 + 语法/迁移彻底性/结构对齐 + Paddle 导入/冒烟检查

## 生成速度优化（在线生成）

- **并行生成**：协议路线的 config 与 handler 两条生成链相互独立，已并行执行（wall-clock 从两者之和降为两者之较长者）
- **检索注入而非全量注入**：协议知识库已从 direct（201 块全量注入每次调用）切换为 semantic（FAISS 向量检索 top_k 注入），提示词缩小约 85%，单次 LLM 调用显著提速；想回到全量注入把 `vendor/herta/data/kb/gdwx/meta.json` 的 `chunk_strategy` 改回 `"direct"` 即可
- **模型选择**：顶栏齿轮可换用更小更快的模型（如 air/flash 档），是剩余最大的速度杠杆；生成质量与速度按演示需要取舍

## 数据与口径

- ONNX 知识库来源：`vendor/onnx/dataset.expanded.jsonl`（pytorch-onnx-dataset2 交付版，1349 条：算子映射 205 / 复合算子实现 347 / 测试用例 327 / 基准测试 219 / 版本兼容矩阵 80 / 转换示例 50 / 技术文档 49 / 错误模式 40 / 修复案例 32；另有 `manifest.json` 记录口径与指纹）
- 星务协议知识库来源：`协议代码数据集2/dataset/augmented`（3090 条增强版：gdwx 253 + dllg 2837，细分块 150 字符/重叠 30，向量随数据集交付），已转换为 herta KB 构建产物格式内置于 `vendor/herta/data/kb/industrial_proto/`；预生成结果与 herta 引擎同目录；PyTorch 样例源码内置于 `vendor/opalign/`（原始出处分别为 `../demo/herta`、`../demo/opalign/EVALUATION`）
- 预生成导出脚本：`server/preset_onnx/*_to_onnx.py`（12 个视觉样例全覆盖；densenet 脚本含 checkpoint 不可 trace 与 MaxPool dilation 不支持两个真实导出难点的处理注释）
- Paddle 已配置并实测 9 个视觉模型：AlexNet、SqueezeNet 1.0、VGG-11、ResNet-18、MNASNet 1.0、ShuffleNet-v2 0.5x、DenseNet-121、Inception-v3、GoogLeNet。`server/paddle_demo_cases.json` 集中维护主模型类、构造参数、输入形状及预期输出；不再因为构造函数需要参数就跳过 VGG/ResNet 的前向检查。
- ModelX 的 Python underlying-library 示例：DenseNet-121 产出 `modified_KaimingNormal.py`，Inception-v3 和 GoogLeNet 产出 `modified_TruncatedNormal.py`。这些伴随模块与主模型一起展示/交付，并一起检查语法和实际导入；界面标记“底层库改写”，可点击查看生成方法。这些是库方法子类改写，不修改已安装 Paddle 文件，不代表已验证 C/CUDA 内核重编译。
- 可运行 `python server/tools/verify_paddle_demos.py --build-presets` 重新执行所有真实转换任务并更新 `server/preset_paddle/*_to_paddle.py` 与相应伴随模块；报告输出到 `artifacts/paddle-demo-validation.json`，记录模型、耗时、文件指纹及真实检查结果。回归要求 `pre_generated=false`，不能把离线回放当作转换成功。
- 知识库页数据集可视化载荷现在按目标独立维护：`vendor/herta/data/kb/baseline_dataset.json`（协议基本数据集 356 条）、`vendor/onnx/dataset_view.json`（ONNX 知识库 1,349 条）和 `vendor/modelx/paddle_dataset_view.json`（ModelX PyTorch→Paddle 映射规则 541 条）；分别用 `server/tools/build_baseline_dataset.py <协议数据集目录>`、`build_onnx_dataset.py <ONNX 数据集目录>`、`build_paddle_dataset.py [ModelX CMC 目录]` 重建
- 算子样例输入为 PyTorch 模型源码（EVALUATION sourceModels，过滤 `__init__` 后 18 个）
- ModelX CMC 转换核心已内置于 `vendor/modelx/CMC/`，首次启用 Paddle 转换前安装 `pip install -r vendor/modelx/requirements.txt`。ModelX 会直接查询已安装的 Paddle API 完成源码转换，因此 PaddlePaddle 是转换依赖，不依赖 LLM；转换完成后服务还会执行一次生成代码的导入/dummy 前向检查
- 本轮 9 模型回归使用 Windows / Python 3.9 / Paddle 2.6.0（CPU 执行）；依赖文件是安装声明，不代表其他 Paddle/Python 版本已通过同等回归。尤其 Python 3.12 的协议环境和 Paddle 3.x 组合尚未在本机验证，现场演示请使用已验证的 Paddle 环境，不要临时升级框架。
- Paddle 自动评测验证的是代码可执行性与静态结构，不是 PyTorch/Paddle 数值等价、数据集精度或推理性能；没有合适构造配置时明确跳过前向，构造/运行异常明确显示失败。
- `vendor/modelx/CMC/` 保留 ModelX 的核心转换器、PyTorch→Paddle 映射表和 AST 匹配器；`server/paddle_converter.py` 只负责把它编排进当前任务系统并汇总转换统计
- 协议评测分数为现场真实重算口径，与历史批次、年度报告综合口径略有差异

## 重新部署（换机器，需有网一次）

**整个平台已自包含：迁移只需拷贝 `domain-codegen/` 一个目录**（引擎、三套知识库数据、演示用例 `case/` 全部内置），对放置位置无要求。

### Linux / macOS / WSL

```bash
conda create -n demo python=3.12 -y && conda activate demo
pip install -e ./vendor/herta        # 引擎与依赖（fastapi/uvicorn/faiss/tree-sitter 等）
pip install python-multipart         # 上传接口需要
pip install torch onnx onnxscript onnxruntime   # ONNX 实导出 + 冒烟测试需要（CPU 版即可；onnxscript 是 torch.onnx 的硬依赖，缺省时回退为仅交付导出脚本）
pip install -r ./vendor/modelx/requirements.txt
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
pip install -r .\vendor\modelx\requirements.txt
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
