"""
inception(v3) PyTorch → ONNX 导出脚本

要点：
1. Inception v3 原生输入尺寸为 299×299（非 224），示例输入需对应调整；
2. 同 GoogLeNet，辅助分类头 aux_logits 必须置 False，避免多余输出分支。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from inception import Inception3  # 源模型定义（与本脚本同目录）


def main():
    model = Inception3(num_classes=1000, aux_logits=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 299, 299)  # Inception v3 原生输入尺寸
    torch.onnx.export(
        model, dummy_input, "inception_v3.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("inception_v3.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: inception_v3.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
