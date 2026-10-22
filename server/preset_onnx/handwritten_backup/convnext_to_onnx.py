"""
convnext PyTorch → ONNX 导出脚本

要点：ConvNeXt 的 LayerScale（通道对角缩放）与 LayerNorm 在 opset>=17
均可表达；GELU 需 opset>=14（本脚本使用 17，无兼容问题）。
block_setting 采用源文件自带的演示配置（tiny 变体）。
"""

import torch
import torch.onnx
import onnx
import onnx.checker

from convnext import ConvNeXt, CNBlockConfig  # 源模型定义（与本脚本同目录）


def main():
    block_setting = [
        CNBlockConfig(input_channels=96, out_channels=192, num_layers=1),
        CNBlockConfig(input_channels=192, out_channels=384, num_layers=1),
        CNBlockConfig(input_channels=384, out_channels=None, num_layers=1),
    ]
    model = ConvNeXt(block_setting=block_setting, num_classes=1000)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, "convnext.onnx",
        opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load("convnext.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"导出完成: convnext.onnx | 节点数={len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
