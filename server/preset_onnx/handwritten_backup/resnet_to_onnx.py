"""
resnet PyTorch → ONNX 导出脚本

以 ResNet-18 结构导出（BasicBlock + [2,2,2,2]）。
残差加法、ReLU、MaxPool 均有 ONNX 直接映射，标准流程即可。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from resnet import ResNet, BasicBlock  # 源模型定义（与本脚本同目录）


def main():
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)  # ResNet-18 结构
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "resnet18.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("resnet18.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: resnet18.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
