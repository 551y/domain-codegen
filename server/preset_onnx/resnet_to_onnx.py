import torch
import onnx

# 假设源模型代码保存在同目录的 resnet.py 文件中
from resnet import ResNet, Bottleneck

def main():
    # 1. 实例化模型 (以 ResNet-50 为例)
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    model.eval()

    # 2. 构造示例输入
    # 注意：使用 dynamo=True 导出时，若需动态 batch 维度，示例输入的 batch size 必须大于 1 (参考知识5)
    # 否则 dynamo 无法正确推断动态维度。此处使用 batch_size=2。
    dummy_input = torch.randn(2, 3, 224, 224, device="cpu")

    onnx_file_path = "resnet50.onnx"

    # 3. 导出模型到 ONNX
    # 使用 dynamo=True 是 PyTorch 2.5+ 推荐的导出方式 (参考知识6)
    # 源码中检出的算子（Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, flatten等）
    # 均为标准算子，在 opset 17 下有原生完美映射，无需自定义 symbolic。
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=True,
    )

    # 4. 校验导出的 ONNX 模型
    onnx_model = onnx.load(onnx_file_path)
    
    # 针对 PyTorch 2.8.0 中可能存在的 allowzero 属性类型错误进行后处理修复 (参考知识2)
    # 该 bug 会导致 allowzero 为布尔值而非整型，引发序列化或校验失败。若已升级至 2.8.1+ 则无需此步。
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == "allowzero":
                # 确保属性类型为 INT，若为其他类型则强制修正为整型 1
                if attr.type != onnx.AttributeProto.INT:
                    attr.type = onnx.AttributeProto.INT
                    attr.i = 1
                    
    onnx.checker.check_model(onnx_model)

    # 5. 打印导出信息
    print(f"模型已成功导出并通过校验：{onnx_file_path}")
    print(f"输入节点: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"输出节点: {[out.name for out in onnx_model.graph.output]}")

if __name__ == "__main__":
    main()