"""
shufflenetv2 PyTorch → ONNX 导出脚本

要点：ShuffleNetV2 的 channel_shuffle 依赖 reshape + transpose 组合
（非独立算子），trace 可正常固化，但需确认目标推理引擎支持
Reshape/Transpose 序列的常量折叠（do_constant_folding=True 已处理）。
x1.0 配置：stages_repeats=[4,8,4], stages_out_channels=[24,116,232,464]。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from shufflenetv2 import ShuffleNetV2  # 源模型定义（与本脚本同目录）


def main():
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464], num_classes=1000)  # x1.0
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "shufflenetv2.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("shufflenetv2.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: shufflenetv2.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
