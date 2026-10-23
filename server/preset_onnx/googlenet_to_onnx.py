import torch
import torch.nn as nn
import onnx

# 假设源模型代码保存在同目录下的 googlenet.py 文件中
from googlenet import GoogLeNet, GoogLeNetOutputs

def main():
    # 1. 实例化模型
    # 设置 aux_logits=False 以避免辅助分支在 eval 模式下输出 None。
    # ONNX 不支持 None 类型的输出张量，关闭辅助分支可保证输出结构稳定。
    model = GoogLeNet(num_classes=1000, aux_logits=False, transform_input=True, init_weights=True)
    model.eval()

    # 数据集源码经剪裁缺失 eager_outputs 方法定义（torchvision 原版在类中定义，
    # 负责把 (logits, aux2, aux1) 封装为 GoogLeNetOutputs namedtuple），forward 中的调用会
    # 触发 AttributeError；这里按输出契约在实例上补挂同名方法
    if not hasattr(model, "eager_outputs"):
        model.eager_outputs = lambda x, aux2, aux1: GoogLeNetOutputs(x, aux2, aux1)

    # 2. 包装模型以处理 namedtuple 输出
    # GoogLeNet 的 forward 返回 GoogLeNetOutputs (namedtuple)，ONNX 导出对 namedtuple 支持有限
    # 这里提取主分支的 logits 作为唯一输出，确保导出为标准的单一张量
    class GoogLeNetONNXWrapper(nn.Module):
        def __init__(self, googlenet_model):
            super().__init__()
            self.model = googlenet_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outputs = self.model(x)
            return outputs.logits

    wrapped_model = GoogLeNetONNXWrapper(model)
    wrapped_model.eval()

    # 3. 构造示例输入
    # 根据【知识 7】，使用 dynamo=True 导出动态 batch 时，示例输入的 batch size 需大于 1，
    # 否则无法正确推断动态维度。这里使用 batch_size = 2。
    # 输入形状为 (batch_size, 3, 224, 224)
    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=False)

    # 4. 定义输入/输出名称和动态轴
    input_names = ["input_image"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_image": {0: "batch_size"},  # 支持动态 batch
        "logits": {0: "batch_size"}
    }

    onnx_file_path = "googlenet.onnx"

    # 注：源码中使用了 torch.nn.functional.adaptive_avg_pool2d 和 nn.AdaptiveAvgPool2d。
    # 根据【知识 6】，在动态形状下某些后端需要将其重写为 avg_pool2d。
    # 但在 PyTorch 的 Dynamo ONNX 导出器中，adaptive_avg_pool2d 会被自动映射为
    # ReduceMean (output_size=1) 或 AveragePool (output_size>1)，原生支持动态 Batch，
    # 因此无需使用 MMDeploy 的自定义重写逻辑。

    # 5. 导出模型到 ONNX
    # 根据【知识 8】，PyTorch 2.5+ 推荐使用 dynamo=True
    print(f"Exporting GoogLeNet to {onnx_file_path} using dynamo exporter...")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=18,  # 使用较新的 opset 以支持所有算子（如 ceil_mode 的 MaxPool 等）
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True  # 推荐的 Dynamo 导出器
        )
    except (AttributeError, Exception) as e:
        # 兼容回退：如果 PyTorch 版本低于 2.5 或 dynamo 导出失败，则使用传统导出器
        print(f"Dynamo export failed with error: {e}. Falling back to legacy exporter.")
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False
        )

    # 6. 校验导出的 ONNX 模型
    print("Validating the exported ONNX model...")
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # 打印模型输入输出信息
    print("\nModel Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Name: {inp.name}, Shape: {shape}")
    
    print("\nModel Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Name: {out.name}, Shape: {shape}")

if __name__ == "__main__":
    main()