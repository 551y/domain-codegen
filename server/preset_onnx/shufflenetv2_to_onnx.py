import torch
import onnx
from onnx import checker

# 假设源模型代码保存在同目录下的 shufflenetv2.py 中
from shufflenetv2 import ShuffleNetV2

def main():
    # 1. 实例化 ShuffleNetV2 模型 (使用标准的 1.0x 配置)
    stages_repeats = [4, 8, 4]
    stages_out_channels = [24, 116, 232, 464, 1024]
    model = ShuffleNetV2(stages_repeats, stages_out_channels, num_classes=1000)
    model.eval()

    # 2. 构造示例输入
    # 【知识 6 修复】dynamo=True 在 batch_size=1 时无法正确推断动态维度。
    # 因此，这里必须使用 batch_size > 1 的示例输入 (例如 2)。
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=False)

    onnx_file_path = "shufflenetv2.onnx"

    # 3. 定义输入/输出名称与动态轴 (支持动态 Batch)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    # 【知识 5 说明】torch.transpose 算子映射：
    # 在 channel_shuffle 中，x 被 view 为 5 维张量后执行 torch.transpose(x, 1, 2)。
    # 因为此时张量的 rank 是静态已知的 (5)，ONNX 导出器可以完美将其映射为带 perm 属性的 ONNX Transpose 算子，
    # 不会触发 "unknown rank" 的 SymbolicValueError，因此无需编写自定义 symbolic。

    # 【知识 3 说明】torch.nn.functional.linear 算子映射：
    # 标准 PyTorch ONNX 导出器会将 2D/3D 输入的 Linear 层映射为高效的 Gemm 算子，
    # 这与 ncnn 需要重写为 MatMul+Add 的场景不同，标准 ONNX Runtime 下无需自定义 rewriter。

    # 4. 导出模型到 ONNX
    # 【知识 7】对于 PyTorch >= 2.5，推荐使用 dynamo=True 导出
    # opset 17 稳定支持所有检出的算子 (Conv, MaxPool, Linear, Transpose, BN, Relu, Cat, Split/Chunk 等)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=True
    )

    # 5. 校验导出的 ONNX 模型并打印信息
    onnx_model = onnx.load(onnx_file_path)
    
    # 【知识 4 规避】检查模型结构，若遇到低版本 PyTorch 导致的 allowzero 布尔类型错误可在此处后处理
    checker.check_model(onnx_model)

    print(f"ONNX model successfully exported and validated to: {onnx_file_path}")
    print("\n--- Model Graph Info ---")
    print(onnx.helper.printable_graph(onnx_model.graph))

if __name__ == "__main__":
    main()