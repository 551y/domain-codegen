"""
mobilenet(v2) PyTorch → ONNX 导出脚本

MobileNetV2 使用默认倒残差结构表（inverted_residual_setting=None 时
构造器自动生成标准配置）；深度可分离卷积在 ONNX 中映射为 Group 卷积。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from mobilenetv2 import MobileNetV2  # 源模型定义（与本脚本同目录）


def main():
    model = MobileNetV2(num_classes=1000)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "mobilenet_v2.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("mobilenet_v2.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: mobilenet_v2.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
