import torch
import onnx
import os

# 假设源文件名为 alexnet.py，从中导入 AlexNet 类
# 为了保证脚本直接可运行，如果导入失败则使用内联定义作为兜底
try:
    from alexnet import AlexNet
except ImportError:
    from typing import Any, Optional
    import torch.nn as nn

    class AlexNet(nn.Module):
        def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


def main():
    # 1. 实例化模型并设置为评估模式（权重随模型加载）
    model = AlexNet(num_classes=1000, dropout=0.5)
    model.eval()

    # 2. 构造示例输入
    # 【知识 5 特殊处理】：使用 dynamo=True 导出时，若需推断动态 batch 维度，示例输入的 batch size 必须大于 1
    # 因此这里使用 batch_size=2 而不是 1，以避免 ExportError
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

    # 3. 导出路径与配置
    onnx_file_path = "alexnet.onnx"
    # 选择 opset 14，稳定支持 Conv, Relu, MaxPool, Flatten, AdaptiveAvgPool 等算子
    opset_version = 14

    input_names = ["input"]
    output_names = ["output"]

    # 配置动态轴，支持动态 batch size
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    # 4. 导出模型到 ONNX
    # 【知识 6 提示】：PyTorch 2.5+ 推荐使用 dynamo=True
    use_dynamo = torch.__version__ >= "2.5.0"

    if use_dynamo:
        print("Using dynamo exporter (PyTorch 2.5+ recommended)...")
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_file_path,
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version
        )
    else:
        print("Using legacy torch.onnx.export exporter...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
        )

    # 5. 后处理：修复知识库中标明的特殊算子属性问题
    # 【知识 2 特殊处理】：PyTorch 2.8.0 可能会错误地将 allowzero 生成布尔值 True 而非整型 1，导致序列化失败
    # 此处进行后处理遍历检查并修正
    onnx_model = onnx.load(onnx_file_path)
    needs_save = False
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == "allowzero":
                # ONNX 规范要求 allowzero 为 int 类型 (attr.i)
                if attr.type == onnx.AttributeProto.FLOAT:
                    attr.i = int(attr.f)
                    attr.type = onnx.AttributeProto.INT
                    needs_save = True
                elif attr.type == onnx.AttributeProto.STRING:
                    attr.i = 1
                    attr.type = onnx.AttributeProto.INT
                    needs_save = True
                    
    if needs_save:
        print("Patched 'allowzero' attribute from boolean to integer type as per Knowledge 2.")
        onnx.save(onnx_model, onnx_file_path)

    # 6. 导出后校验模型
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")

    # 7. 打印导出信息
    print(f"Model successfully exported to: {os.path.abspath(onnx_file_path)}")
    print(f"Input names: {input_names}, Dynamic axes: {dynamic_axes['input']}")
    print(f"Output names: {output_names}, Dynamic axes: {dynamic_axes['output']}")
    print(f"Opset version: {opset_version}")


if __name__ == "__main__":
    main()