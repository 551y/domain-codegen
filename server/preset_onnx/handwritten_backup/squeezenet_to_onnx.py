"""
squeezenet PyTorch → ONNX 导出脚本

SqueezeNet 为极致轻量骨干（Fire 模块：squeeze 1x1 + expand 1x1/3x3），
适合机载算力受限场景；全部为标准卷积算子，直接映射导出。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from squeezenet import SqueezeNet  # 源模型定义（与本脚本同目录）


def main():
    model = SqueezeNet(version="1_0", num_classes=1000)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "squeezenet.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("squeezenet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: squeezenet.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
