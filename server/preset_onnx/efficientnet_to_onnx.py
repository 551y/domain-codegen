import torch
import onnx
from onnx import checker
import os

# 假设源模型代码保存在同目录下的 efficientnet_model.py
from efficientnet_model import EfficientNet, MBConvConfig, FusedMBConvConfig

def export_efficientnet_to_onnx():
    # 1. 实例化模型
    # 构造一个轻量级的 EfficientNet 配置用于演示导出链路
    inverted_residual_setting = [
        MBConvConfig(1, 3, 1, 32, 16, 1),
        MBConvConfig(6, 3, 2, 16, 24, 1),
    ]
    
    model = EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.2,
        stochastic_depth_prob=0.2,
        num_classes=1000
    )
    
    # 必须切换到 eval 模式
    # 这会让 Dropout 和 StochasticDepth 算子在推理模式下工作，避免导出随机数相关的训练图
    model.eval()

    # 2. 构造示例输入
    # 【关键处理参考知识5】：使用 dynamo=True 导出时，若 batch_size=1 可能无法正确推断动态维度。
    # 因此这里使用 batch_size=2 的示例输入，以确保动态 batch 轴被正确创建。
    batch_size = 2
    channels = 3
    height, width = 224, 224
    dummy_input = torch.randn(batch_size, channels, height, width)

    # 3. 定义导出路径和动态轴
    onnx_file_path = "efficientnet_model.onnx"
    
    # 定义动态 batch 维度
    dynamic_axes = {
        'input': {0: 'batch_size'},    # 第 0 维是 batch
        'output': {0: 'batch_size'}    # 输出的第 0 维也是 batch
    }

    # 4. 执行 ONNX 导出
    # 【关键处理参考知识6】：PyTorch >= 2.5 推荐使用 dynamo=True，它基于 torch.export，对复杂算子（如 StochasticDepth, Permute）的 ATen 映射更准确。
    # 【算子映射说明】：
    # - StochasticDepth: 在 eval 模式下退化为恒等映射，dynamo 追踪时会自动将其消除或映射为 ONNX Identity，无需自定义 symbolic。
    # - torch.permute: 会被 dynamo 自动映射为 ONNX Transpose 算子（参考知识7）。
    # - torch.flatten: 会被映射为 ONNX Flatten 或 Reshape 算子（参考知识4）。
    # - nn.Dropout: 在 eval 模式下被自动移除，不产生 ONNX Dropout 节点（参考知识8）。
    print(f"Starting ONNX export with dynamo=True...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=17,        # 使用较新的 opset 以支持所有现代算子
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        dynamo=True              # 启用推荐的 Dynamo 导出器
    )
    print(f"Model successfully exported to {onnx_file_path}")

    # 5. 后处理：修复 allowzero 属性的布尔值问题
    # 【关键处理参考知识2】：PyTorch 2.8.0 中可能错误地将 allowzero 属性序列化为布尔值而非整型，导致部分推理引擎加载失败。
    # 这里加入后处理逻辑，确保属性类型合规。
    print("Post-processing ONNX model to fix potential 'allowzero' attribute type bug...")
    onnx_model = onnx.load(onnx_file_path)
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == 'allowzero':
                # 确保如果存在 allowzero 属性，其类型为 int
                if attr.type == onnx.AttributeProto.INT and attr.i == 1:
                    pass # 已经是正确的整型
                elif attr.type == onnx.AttributeProto.FLOAT or attr.type == onnx.AttributeProto.STRING:
                    # 兼容性修复：如果被错误写入，强制转换回 int
                    attr.i = 1
                    attr.type = onnx.AttributeProto.INT
    onnx.save(onnx_model, onnx_file_path)
    print("Post-processing completed.")

    # 6. 导出校验与信息打印
    print("Validating ONNX model...")
    try:
        # 使用 onnx.checker 校验模型的形状和算子合法性
        onnx_model = onnx.load(onnx_file_path)
        checker.check_model(onnx_model)
        print("ONNX model validation passed!")
        
        # 打印模型基本信息
        print("\n--- ONNX Model Info ---")
        print(f"IR Version: {onnx_model.ir_version}")
        print(f"Opset Import: {[op.version for op in onnx_model.opset_import]}")
        print(f"Producer Name: {onnx_model.producer_name}")
        
        # 打印输入输出信息
        for inp in onnx_model.graph.input:
            print(f"Input: {inp.name}, Shape: {[d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]}")
        for out in onnx_model.graph.output:
            print(f"Output: {out.name}, Shape: {[d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]}")
            
    except Exception as e:
        print(f"ONNX model validation failed: {e}")

if __name__ == "__main__":
    export_efficientnet_to_onnx()