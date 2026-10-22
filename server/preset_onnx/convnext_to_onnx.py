import os
import sys
import collections
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import onnx

# ==========================================
# 1. 模型定义 (整合自源码，去除相对导入以保证独立运行)
# ==========================================

def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple([x] * n)

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Optional[Union[int, Tuple[int, ...], str]] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: Union[int, Tuple[int, ...]] = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
            conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
            
        super().__init__(*layers)
        self.out_channels = out_channels

class Conv2dNormActivation(ConvNormActivation):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Optional[Union[int, Tuple[int, int], str]] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: Union[int, Tuple[int, int]] = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            groups, norm_layer, activation_layer, dilation, inplace, bias, torch.nn.Conv2d,
        )

class Permute(torch.nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)

def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

class CNBlockConfig:
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=4, stride=4, padding=0,
                norm_layer=norm_layer, activation_layer=None, bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# ==========================================
# 2. ONNX 导出逻辑
# ==========================================

def export_convnext_to_onnx(onnx_file_path: str = "convnext.onnx"):
    # 1. 实例化模型
    block_setting = [
        CNBlockConfig(input_channels=96, out_channels=192, num_layers=1),
        CNBlockConfig(input_channels=192, out_channels=384, num_layers=1),
        CNBlockConfig(input_channels=384, out_channels=None, num_layers=1)
    ]
    model = ConvNeXt(block_setting=block_setting, num_classes=1000)
    
    # 必须设置为 eval 模式，否则 StochasticDepth 的随机噪声生成逻辑会被追踪，导致导出失败或推理结果错误
    model.eval()

    # 2. 构造 dummy input
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, channels, height, width, dtype=torch.float32)

    # 3. 定义输入输出名与动态轴 (支持动态 batch)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size"},   # 第 0 维动态
        "output": {0: "batch_size"}   # 第 0 维动态
    }

    # 4. 导出 ONNX
    # 使用 opset_version=20 以原生支持 GELU 算子 (参考知识库：Opset 20 包含 Gelu 映射，避免低版本不支持的问题)
    print(f"Exporting model to ONNX at {onnx_file_path} with opset 20...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    print("Export finished.")

    # 5. 后处理与校验
    # 参考知识库：PyTorch 2.8.0 存在 allowzero 属性导出为布尔值而非整数的 bug，此处加入后处理兼容逻辑
    onnx_model = onnx.load(onnx_file_path)
    
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.name == "allowzero":
                # ONNX 规范要求 allowzero 为 int 类型，若被错误序列化为 bool 则修正
                if attr.type == onnx.AttributeProto.INT and attr.i == 1:
                    pass # 正常情况
                else:
                    # 强制修正为整数 1
                    attr.type = onnx.AttributeProto.INT
                    attr.i = 1
                    
    onnx.save(onnx_model, onnx_file_path)

    # 使用 onnx.checker 校验模型合法性
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")

    # 打印导出模型的基本信息
    print("\n--- ONNX Model Info ---")
    print(f"IR version: {onnx_model.ir_version}")
    print(f"Opset version: {onnx_model.opset_import[0].version}")
    print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"Outputs: {[outp.name for outp in onnx_model.graph.output]}")

if __name__ == "__main__":
    export_convnext_to_onnx()