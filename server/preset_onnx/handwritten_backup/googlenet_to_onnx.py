"""
googlenet PyTorch → ONNX 导出脚本

要点：Inception 系列带辅助分类头（aux_logits），仅训练阶段使用；
导出推理图时必须置 aux_logits=False，否则 ONNX 会多出一路辅助输出。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from googlenet import GoogLeNet  # 源模型定义（与本脚本同目录）


def main():
    model = GoogLeNet(num_classes=1000, aux_logits=False)  # 关闭辅助分类头
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "googlenet.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("googlenet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: googlenet.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
