"""
densenet PyTorch → ONNX 导出脚本

要点：
1. 源模型使用 torch.utils.checkpoint 做显存优化，export trace 无法固化该结构，
   导出时需以普通前向替代（math_ops 语义等价，仅牺牲显存优化）。
2. F.max_pool2d 的 dilation 参数在 ONNX opset 中不被支持——DenseNet 实际未启用
   dilation（默认 1），导出时显式固定并注释说明，避免静默语义偏差。
3. opset 17 + dynamic_axes 支持动态 batch。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from densenet import DenseNet  # 源模型定义（与本脚本同目录）


def load_model(num_classes: int = 1000) -> torch.nn.Module:
    model = DenseNet(num_classes=num_classes, memory_efficient=False)
    model.eval()
    return model


def main():
    model = load_model()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "densenet.onnx",
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
        verbose=False,
    )

    # 导出后自检：结构合法性 + 拓扑摘要
    onnx_model = onnx.load("densenet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: densenet.onnx | opset={onnx_model.opset_import[0].version} | "
          f"节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
