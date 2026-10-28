# SemTrans 前后端真实链路测试

测试方式：启动 FastAPI，调用 `/api/generate`，每 0.8 秒轮询 `/api/task/{task_id}`，记录 `evaluation_steps` 状态变化；前端轮询周期为 1.5 秒。

当前活动服务：ZCode / `glm-5.3`。测试没有伪造结果。

| 模型 | 后端结果 | 耗时 | 阶段结论 |
|---|---|---:|---|
| AlexNet | `completed`, `verified=true` | 70.0 s | 准备、候选生成、语义对齐、独立复验均按动态状态完成 |
| DenseNet | `failed` | 155.2 s | 候选生成阶段等待 ZCode 超时；未进入 SemTrans 评测 |
| ShuffleNetV2（修复后） | `completed`, `verified=true` | 129.3 s | 候选生成、语义对齐、独立复验均通过 |
| SqueezeNet 1.0（新增） | `completed`, `verified=true` | 40.7 s | 候选生成、语义对齐、独立复验均通过 |
| MNASNet（新增） | `completed`, `verified=true` | 43.3 s | 残差显式化后，候选生成、语义对齐、独立复验均通过 |

AlexNet 实际状态序列为：

```text
准备 running
准备 passed → 候选生成 running
候选生成 passed → 语义对齐 running
语义对齐 passed → 独立复验 running
独立复验 passed → completed
```

修复前的 DenseNet/ShuffleNetV2 失败状态也曾由后端动态写入，前端正确显示红色失败阶段；修复后 ShuffleNetV2 已通过真实后端链路。

前端 `npm run build`：通过。

补充：之前用 DeepSeek `deepseek-chat` 单独跑 DenseNet 时，候选生成约 5.5 秒，完整 SemTrans 流程约 88.9 秒并通过；因此 DenseNet 当前问题主要是 ZCode `glm-5.3` 长响应，而不是 DenseNet frontend 本身无法转换。
