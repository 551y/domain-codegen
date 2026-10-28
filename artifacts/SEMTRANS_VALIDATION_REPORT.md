# SemTrans 真实模型验收报告

日期：2026-10-29（Asia/Shanghai）

## 当前结论

- **AlexNet：通过。** 真实 PyTorch/Paddle frontend、SemTrans 语义对齐和独立输入复验均通过。
- **DenseNet：通过。** 真实模型 SemTrans 本地端到端回归通过；DeepSeek 路线也曾完整通过。
- **ShuffleNetV2：通过。** 修复 `torch.chunk()` tuple 输出保存后，真实 SemTrans 对齐和独立复验通过；真实后端 API 任务也通过。
- **SqueezeNet 1.0：通过。** 新增真实模型 profile，DeepSeek 后端 API 任务完成，语义对齐和独立复验通过。
- **MNASNet：通过。** Paddle 候选生成阶段将残差表达式规范化为显式 `paddle.add`，Paddle frontend 能保留残差图边，SemTrans 语义对齐和独立复验均通过。

## ShuffleNetV2 修复

问题不是模型算子数学定义错误，而是 SemTrans tracer 在缓存单个图节点的多输出时只保留了第一个分支：

```text
torch.chunk(x, 2) -> (branch0, branch1)
```

后续 `getitem` 需要读取 `branch1`，旧逻辑却把 `chunk` 缓存成 `branch0`，导致第二个 inverted residual block 的输入被错误定位。现在 tuple-valued 单图节点会保留全部分支，后续 `getitem` 能正确选择分支，并新增回归测试。

## 实测结果

| 模型 | 本地 SemTrans 回归 | 真实后端任务 |
|---|---|---|
| AlexNet | 通过 | 通过，约 70 秒 |
| DenseNet | 通过 | 当前 ZCode `glm-5.3` 候选生成可能超时；DeepSeek 路线约 93 秒通过 |
| ShuffleNetV2 | 通过 | 通过，候选生成约 105.5 秒，总耗时约 127.9 秒 |
| SqueezeNet 1.0 | 通过 | DeepSeek 后端约 40.7 秒通过 |
| MNASNet | 通过 | DeepSeek 后端约 43.3 秒通过 |

ShuffleNetV2 后端阶段按 `running → passed → running → passed → running → passed → completed` 动态推进，前端会逐阶段显示，不是初始全绿。

## 验证命令

- `server/tools/verify_real_alexnet.py`
- `server/tools/verify_real_models.py`
- `server/tests/test_semtrans_components.py`
- 前端 `npm run build`

所有报告和截图均不包含 API Key。
