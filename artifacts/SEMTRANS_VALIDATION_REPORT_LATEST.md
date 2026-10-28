# SemTrans 真实模型验收报告

日期：2026-10-29（Asia/Shanghai）

## 当前结论

- **AlexNet：通过。** 使用真实 bundled PyTorch 源码、真实 Paddle 目标实现，SemTrans 语义对齐和独立输入复验均通过；在线 ZCode 路线也已实际通过。
- **DenseNet：通过离线端到端回归。** 真实 PyTorch/Paddle frontend、参数注入、SemTrans-JIT 对齐和数值复验均通过。
- **ShuffleNetV2：已通过。** 修复 channel-shuffle 中 tuple-valued `torch.chunk()` 输出被压成单分支的问题后，真实 SemTrans 语义对齐和独立复验均通过。
- **MNASNet：已通过。** 生成候选中的残差表达式统一为显式 `paddle.add`，使 Paddle frontend 保留残差图边；真实 SemTrans 对齐和独立复验通过。

## 修复内容

1. Paddle Linear 的语义定义补充 `weight: W` 和 `bias: B` 参数映射，并保留 `bias_attr` 作为 bias 开关。
2. SemTrans 参数注入识别 PyTorch `[out, in]` 到 Paddle `[in, out]` 的 Linear 权重转置，包括方阵权重。
3. 语义诊断将 Paddle `ba` 归一化为统一的 `bias` 绑定。
4. 动态 shape bookkeeping、Torch `view` 与 Paddle `reshape` 在 channel-shuffle 区域使用统一执行路径；形状常量不会再被当成模型参数注入。
5. 保留 tuple-valued 单图节点（例如 `torch.chunk()`）的全部分支，后续 `getitem` 可以正确选择第二分支；增加对应回归测试。
6. 产品评测使用显式跨框架 FP32 容差 `atol=rtol=1e-3`，该值会返回在评测报告中；ONNX 路线不变。

## 现场演示口径

当前 AlexNet、DenseNet、ShuffleNetV2、SqueezeNet 和 MNASNet 均已通过本地 SemTrans 真实模型回归；MNASNet 使用 DeepSeek 的真实后端任务也已通过，总耗时约 43.3 秒。DenseNet 在线 ZCode 候选生成仍可能受 `glm-5.3` 长响应影响。

报告和截图不包含 API Key。
