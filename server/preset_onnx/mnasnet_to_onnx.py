import torch
import onnx
from mnasnet import MNASNet  # 假设源模型代码保存在同目录下的 mnasnet.py 中

def export_mnasnet_to_onnx():
    # 1. 实例化模型并设置为评估模式
    # 使用默认的 alpha=1.0, num_classes=1000, dropout=0.2
    model = MNASNet(alpha=1.0, num_classes=1000, dropout=0.2)
    model.eval()

    # 2. 构造示例输入
    # 知识库中指出：使用 dynamo=True 导出时，若 batch_size=1 会导致动态维度推断失败。
    # 为保证兼容性及动态轴正确推断，此处使用 batch_size=2 构造示例输入。
    # 标准 ImageNet 输入尺寸为 (B, C, H, W) -> (2, 3, 224, 224)
    dummy_input = torch.randn(2, 3, 224, 224, dtype=torch.float32)

    # 3. 定义导出参数
    onnx_file_path = "mnasnet.onnx"
    input_names = ["input"]
    output_names = ["output"]
    
    # 设置动态轴，支持动态 Batch 推理
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    # 4. 执行 ONNX 导出
    # 算子处理说明：
    # 本模型包含的算子有：Conv2d, BatchNorm2d, ReLU, Dropout, Linear, 以及 x.mean([2, 3]) (对应 ONNX ReduceMean)。
    # 这些算子在 ONNX opset 14 中均有原生且稳定的标准映射（如知识库中 Relu 映射为 onnx::Relu），
    # 因此无需额外编写自定义 symbolic (torch.onnx.register_custom_op_symbolic)。
    # 选择 opset_version=14 可确保所有上述算子的精确映射与兼容性。
    print("Exporting MNASNet to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,        # 导出模型权重
        opset_version=14,          # 选择 opset 14，原生支持上述所有算子
        do_constant_folding=True,  # 常量折叠优化
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"Model successfully exported to {onnx_file_path}")

    # 5. 校验导出的 ONNX 模型
    print("Validating the ONNX model...")
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # 6. 打印导出的模型图信息
    print("\n--- ONNX Model Graph Info ---")
    print(onnx.helper.printable_graph(onnx_model.graph))

if __name__ == "__main__":
    export_mnasnet_to_onnx()