import torch
import onnx
from onnx import AttributeProto
# 假设源模型代码保存在同目录下的 squeezenet.py 中
from squeezenet import SqueezeNet


def export_squeezenet_to_onnx():
    # 1. 实例化模型并设置为评估模式
    # 在 eval 模式下，nn.Dropout 会被 ONNX 导出器自动映射为恒等算子，无需自定义 symbolic
    model = SqueezeNet(version="1_0", num_classes=1000, dropout=0.5)
    model.eval()

    # 2. 构造合理的示例输入 (Batch, Channel, Height, Width)
    # SqueezeNet 标准输入尺寸为 224x224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # 3. 定义导出路径和输入输出节点名称
    onnx_file_path = "squeezenet.onnx"
    input_names = ["input"]
    output_names = ["output"]

    # 4. 定义动态轴，支持动态 Batch Size 推理
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    # 5. 执行 ONNX 导出
    # 选择 opset_version=14，因为：
    # - ONNX opset 12+ 原生支持 MaxPool 的 ceil_mode 属性，无需自定义 symbolic
    # - opset 14 完善了 Flatten 和 Relu 的类型推导与映射
    print("Starting ONNX export...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,        # 权重随模型导出
        opset_version=14,          # 指定 opset 版本以支持 ceil_mode 等属性
        do_constant_folding=True,  # 开启常量折叠优化
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"Model successfully exported to {onnx_file_path}")

    # 6. 后处理：修复可能存在的 allowzero 属性类型错误
    # 参考 ONNX 知识库：部分 PyTorch 版本导出 Flatten/Reshape 时会错误地将 allowzero 生成为 bool 而非 int
    onnx_model = onnx.load(onnx_file_path)
    needs_save = False
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == "allowzero":
                # ONNX 规范要求 allowzero 必须是 int 类型 (0 或 1)
                if attr.type == AttributeProto.FLOAT:
                    attr.i = int(attr.f)
                    attr.type = AttributeProto.INT
                    del attr.f
                    needs_save = True
                elif attr.type == AttributeProto.STRING and attr.s in (b'True', b'1'):
                    attr.i = 1
                    attr.type = AttributeProto.INT
                    del attr.s
                    needs_save = True

    if needs_save:
        onnx.save(onnx_model, onnx_file_path)
        print("Patched 'allowzero' attribute type from boolean to integer.")

    # 7. 使用 onnx.checker 校验导出的模型
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")

    # 8. 打印导出模型的关键信息
    print("\n--- ONNX Model Info ---")
    print(f"IR version: {onnx_model.ir_version}")
    print(f"Opset version: {onnx_model.opset_import[0].version}")
    print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"Outputs: {[out.name for out in onnx_model.graph.output]}")
    
    for inp in onnx_model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)  # 动态维度
            else:
                shape.append(d.dim_value)  # 静态维度
        print(f"Shape of '{inp.name}': {shape}")


if __name__ == "__main__":
    export_squeezenet_to_onnx()