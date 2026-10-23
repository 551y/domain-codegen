import torch
import onnx
from onnx import AttributeProto

# 假设源文件名为 vgg.py，从中导入模型定义
from vgg import VGG, make_layers, cfgs


def main():
    # 1. 实例化 VGG 模型 (使用 VGG-16 配置 'D')
    # 设置 init_weights=True 以应用 kaiming_normal_ 等初始化
    model = VGG(make_layers(cfgs["D"]), num_classes=1000, init_weights=True)
    model.eval()  # 切换到评估模式，Dropout 将被忽略或作为恒等映射导出

    # 2. 构造示例输入
    # 根据 Knowledge 5，dynamo 导出器需要 batch size > 1 来正确推断动态维度
    batch_size = 2
    example_input = torch.randn(batch_size, 3, 224, 224, requires_grad=False)

    # 3. 定义 ONNX 导出路径和输入输出名称
    onnx_file_path = "vgg16.onnx"
    input_names = ["input_image"]
    output_names = ["output_logits"]
    
    # 定义动态轴，支持动态 batch
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "output_logits": {0: "batch_size"}
    }

    # 4. 导出模型到 ONNX
    # 根据 Knowledge 6，推荐使用 dynamo=True (PyTorch >= 2.5)
    # 所有检出的算子（Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, flatten, Linear, Dropout）
    # 在 opset 17 中均有原生映射，无需自定义 symbolic
    print(f"正在导出模型到 {onnx_file_path} ...")
    torch.onnx.export(
        model,
        example_input,
        onnx_file_path,
        export_params=True,       # 导出模型权重
        opset_version=17,         # 使用现代 opset 版本
        do_constant_folding=True, # 执行常量折叠优化
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=True               # 推荐的 Dynamo 导出器
    )

    # 5. 校验导出的 ONNX 模型
    print("正在校验导出的 ONNX 模型 ...")
    onnx_model = onnx.load(onnx_file_path)
    
    # 根据 Knowledge 2，部分 PyTorch 版本可能错误地将 allowzero 属性生成为布尔值
    # 这里进行后处理修复，确保其类型为 INT（新版 onnx 枚举无 BOOL，按 FLOAT/STRING 异常类型识别）
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == "allowzero" and attr.type != AttributeProto.INT:
                if attr.type == AttributeProto.FLOAT:
                    attr.i = int(attr.f)
                    attr.ClearField("f")
                else:
                    attr.i = 1
                    attr.ClearField("s")
                attr.type = AttributeProto.INT
    
    onnx.save(onnx_model, onnx_file_path)
    
    # 重新加载修复后的模型进行校验
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    
    # 6. 打印导出信息
    print("ONNX 模型导出并校验成功！")
    print(f"模型文件路径: {onnx_file_path}")
    print(f"输入节点: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"输出节点: {[out.name for out in onnx_model.graph.output]}")


if __name__ == "__main__":
    main()