"""
efficientnet PyTorch → ONNX 导出脚本

要点：EfficientNet 的 SiLU（swish）激活在 opset>=14 原生支持；
SE 模块的全局均值池化映射为 GlobalAveragePool。
inverted_residual_setting 采用轻量演示配置（两阶段 MBConv）。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from efficientnet import EfficientNet, MBConvConfig  # 源模型定义（与本脚本同目录）


def main():
    setting = [
        MBConvConfig(expand_ratio=1, kernel=3, stride=1, input_channels=32, out_channels=16, num_layers=1),
        MBConvConfig(expand_ratio=6, kernel=3, stride=2, input_channels=16, out_channels=24, num_layers=2),
    ]
    model = EfficientNet(setting, dropout=0.2, num_classes=1000)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "efficientnet.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("efficientnet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: efficientnet.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
