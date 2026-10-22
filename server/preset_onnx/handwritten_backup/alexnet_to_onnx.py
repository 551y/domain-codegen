"""
alexnet PyTorch → ONNX 导出脚本

AlexNet 结构规整（Conv/ReLU/MaxPool/Dropout/Linear），无特殊算子，
标准 torch.onnx.export 即可；Dropout 在推理模式下自动消除（eval 态 trace）。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from alexnet import AlexNet  # 源模型定义（与本脚本同目录）


def main():
    model = AlexNet(num_classes=1000)
    model.eval()  # eval 态导出：Dropout/BatchNorm 进入推理语义

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "alexnet.onnx",
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("alexnet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: alexnet.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
