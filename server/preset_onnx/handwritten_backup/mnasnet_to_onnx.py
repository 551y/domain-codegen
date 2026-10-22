"""
mnasnet PyTorch → ONNX 导出脚本

MNASNet 为神经架构搜索得到的轻量骨干（倒残差 + SE 注意力），
SE 分支的 mul/sigmoid 均有 ONNX 映射；alpha=1.0 为标准宽度。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from mnasnet import MNASNet  # 源模型定义（与本脚本同目录）


def main():
    model = MNASNet(alpha=1.0, num_classes=1000)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "mnasnet.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("mnasnet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: mnasnet.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
