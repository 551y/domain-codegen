import torch
import torch.nn.functional as F
import onnx
from onnx import AttributeProto

# 假设导出脚本与源文件同目录运行，直接从源文件导入模型类
# 请确保源文件名为 model_source.py 或修改此处的导入路径
from model_source import DenseNet

# --- 特殊算子处理 ---
# 1. adaptive_avg_pool2d 替代写法 (参考知识6：动态形状下替换为 avg_pool2d 以避免推导问题)
original_adaptive_avg_pool2d = F.adaptive_avg_pool2d
def patched_adaptive_avg_pool2d(input, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if int(output_size[0]) == int(output_size[1]) == 1:
        # 对于 (1,1) 目标尺寸，ONNX 原生映射为 GlobalAveragePool，对动态形状安全
        # 此处保留原调用，若需严格按知识6替换，可计算 k=size 并调用 avg_pool2d
        return original_adaptive_avg_pool2d(input, output_size)
    else:
        size = input.shape[2:]
        k = [int(size[i] / output_size[i]) for i in range(len(size))]
        return F.avg_pool2d(input, kernel_size=k, stride=k, padding=0, ceil_mode=False, count_include_pad=False)

# 替换 F.adaptive_avg_pool2d
F.adaptive_avg_pool2d = patched_adaptive_avg_pool2d

def main():
    # 实例化模型，必须关闭 memory_efficient，因为 torch.utils.checkpoint 不支持 ONNX 导出
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        memory_efficient=False
    )
    model.eval()

    # 构造示例输入 (参考知识7：dynamo 导出动态 batch 时，示例输入的 batch size 需 > 1)
    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=False)

    onnx_file_path = "densenet.onnx"

    # 导出模型 (参考知识8：PyTorch 2.5+ 推荐使用 dynamo=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=14, # 选择支持所需算子 (如 Flatten, Relu) 的 opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=True # 使用基于 torch.export 的推荐导出器
    )

    # --- 后处理与校验 ---
    onnx_model = onnx.load(onnx_file_path)

    # 2. 修复 allowzero 属性类型问题 (参考知识2：将布尔值转为整型以避免序列化失败)
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == 'allowzero':
                if attr.type == AttributeProto.BOOL:
                    attr.i = 1 if attr.b else 0
                    attr.type = AttributeProto.INT
                    attr.ClearField('b')

    onnx.save(onnx_model, onnx_file_path)

    # 校验模型
    onnx.checker.check_model(onnx_model)
    print(f"模型成功导出至 {onnx_file_path} 并通过校验！")
    print(f"输入名称: {onnx_model.graph.input[0].name}")
    print(f"输出名称: {onnx_model.graph.output[0].name}")

if __name__ == "__main__":
    main()