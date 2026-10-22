"""
vgg PyTorch → ONNX 导出脚本

VGG 为纯 Conv/ReLU/MaxPool/Linear 堆叠，算子均有 ONNX 直接映射；
注意自适应平均池化(nn.AdaptiveAvgPool2d)在 opset>=11 原生支持，无需改写。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from vgg import VGG  # 源模型定义（与本脚本同目录）


def main():
    model = VGG(num_classes=1000)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "vgg.onnx",
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("vgg.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: vgg.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
