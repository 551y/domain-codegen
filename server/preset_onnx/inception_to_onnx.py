import torch
import torch.nn.functional as F
import onnx
from onnx import helper, AttributeProto

# 假设源模型代码保存为同目录下的 model_source.py
from model_source import Inception3

# 定义一个包装器，确保模型只返回主 logits 张量，便于 ONNX 导出和推理部署
class InceptionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inception3 默认返回 namedtuple，这里提取主输出
        return self.model(x).logits


if __name__ == "__main__":
    # 1. 实例化模型并设置为评估模式
    # 设置 aux_logits=False 关闭辅助分类器，init_weights=True 抑制警告
    model = Inception3(num_classes=1000, aux_logits=False, init_weights=True)
    model = InceptionWrapper(model)
    model.eval()

    # 2. 构造示例输入
    # Batch size 设为 2 (>1)，避免 dynamo=True 导出时动态维度推断失败的问题
    # InceptionV3 标准输入尺寸为 3 x 299 x 299
    dummy_input = torch.randn(2, 3, 299, 299, device="cpu")

    # 3. 针对 adaptive_avg_pool2d 的自定义替代写法
    # 参考 MMDeploy 知识库：当 output_size != (1,1) 时，部分 ONNX 后端不支持，
    # 显式替换为 avg_pool2d 以提升兼容性；当 output_size=(1,1) 时保持原样以映射到 GlobalAveragePool
    origin_adaptive_avg_pool2d = F.adaptive_avg_pool2d
    def patched_adaptive_avg_pool2d(input, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if int(output_size[0]) == int(output_size[1]) == 1:
            return origin_adaptive_avg_pool2d(input, output_size)
        else:
            size = input.shape[2:]
            k = [int(size[i] / output_size[i]) for i in range(len(size))]
            return F.avg_pool2d(
                input, kernel_size=k, stride=k, padding=0, 
                ceil_mode=False, count_include_pad=False
            )
    
    # 应用 Monkey-patch
    F.adaptive_avg_pool2d = patched_adaptive_avg_pool2d

    onnx_path = "inception_v3.onnx"

    # 4. 导出 ONNX 模型
    print("Starting ONNX export...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14, # 选用 14 保证算子兼容性
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        do_constant_folding=True,
        verbose=False
    )
    print(f"Export finished. Model saved to {onnx_path}")

    # 恢复原始的 adaptive_avg_pool2d，避免影响其他代码
    F.adaptive_avg_pool2d = origin_adaptive_avg_pool2d

    # 5. 后处理：修复 allowzero 属性类型问题
    # 参考 PyTorch Issue 161941：部分版本错误地生成了 boolean True 而非 integer 1
    print("Applying post-processing fixes (allowzero attribute)...")
    onnx_model = onnx.load(onnx_path)
    for node in onnx_model.graph.node:
        attrs_to_fix = []
        for i, attr in enumerate(node.attribute):
            if attr.name == "allowzero" and attr.type == AttributeProto.BOOL:
                # 构造整型的 allowzero 属性替换布尔型
                new_attr = helper.make_attribute("allowzero", 1)
                attrs_to_fix.append((i, attr, new_attr))
        
        # 倒序修改以维持索引正确性
        for i, old_attr, new_attr in reversed(attrs_to_fix):
            node.attribute.remove(old_attr)
            node.attribute.insert(i, new_attr)
            
    onnx.save(onnx_model, onnx_path)
    print("Post-processing finished.")

    # 6. 校验模型并打印导出信息
    print("Validating ONNX model...")
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

    print("\n--- ONNX Model Export Info ---")
    print(f"Model File: {onnx_path}")
    print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"Outputs: {[out.name for out in onnx_model.graph.output]}")
    
    # 解析并打印动态维度信息
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    print("Input Dynamic Axes:")
    for dim in input_shape.dim:
        print(f"  - {dim.dim_param if dim.dim_param else dim.dim_value}")