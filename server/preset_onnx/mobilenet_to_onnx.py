import torch
import onnx
from onnx import checker

# 假设导出脚本与源文件同目录运行，直接从源文件导入模型类
from mobilenetv2 import MobileNetV2
# from mobilenetv3 import MobileNetV3 # MobileNetV3需要额外的InvertedResidualConfig配置，此处以MobileNetV2为例

def export_mobilenet_to_onnx():
    # 1. 实例化模型并设置为评估模式
    model = MobileNetV2(num_classes=1000)
    model.eval()

    # 2. 构造示例输入
    # 【知识 8 修复】使用 batch_size > 1 (例如 2) 的示例输入，避免 dynamo 导出时无法推断动态维度的问题
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=False)

    # 3. 定义导出参数
    onnx_file_path = "mobilenetv2.onnx"
    input_names = ["input"]
    output_names = ["output"]

    # 支持动态 batch 维度
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    # 4. 执行 ONNX 导出
    # 【知识 3 映射】Hardswish 算子在 opset 14+ 原生映射为 ONNX 的 HardSwish 算子，无需自定义 symbolic
    # 【知识 5 映射】torch.flatten 原生映射为 ONNX 的 Flatten 算子
    # 【知识 6 映射】adaptive_avg_pool2d 输出尺寸为 (1,1) 时，原生导出会映射为 GlobalAveragePool，对动态形状友好
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,        # 权重随模型导出
        opset_version=14,          # 选择 opset 14 以原生支持 HardSwish
        do_constant_folding=True,  # 执行常量折叠优化
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # 5. 校验导出的 ONNX 模型
    onnx_model = onnx.load(onnx_file_path)
    checker.check_model(onnx_model)

    # 6. 打印导出信息
    print(f"模型已成功导出至: {onnx_file_path}")
    print("ONNX 模型校验通过！")
    print("\n--- ONNX 模型图信息 ---")
    print(f"输入节点: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"输出节点: {[out.name for out in onnx_model.graph.output]}")

if __name__ == "__main__":
    export_mobilenet_to_onnx()