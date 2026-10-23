# PyTorch → PaddlePaddle 转换产物（模型部署适配知识库增强生成）
# 对齐说明：
#   - torch.cat(xs, 1) → paddle.concat(xs, axis=1)：API 改写 + 参数变换 dim → axis
#   - torch.nn.MaxPool2d(ceil_mode=True) → paddle.nn.MaxPool2D(ceil_mode=True)：同名参数语义一致
#   - torch.nn.init.kaiming_uniform_ / normal_ / constant_ → paddle.nn.initializer 等效分布，
#     以 set_value 在构造后重写（torch 为原地初始化函数，Paddle 为初始化策略对象）
import math
from typing import Any, Optional

import paddle
import paddle.nn as nn

__all__ = ["SqueezeNet"]


class Fire(nn.Layer):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2D(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2D(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2D(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        # torch.cat(..., 1) → paddle.concat(..., axis=1)
        return paddle.concat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], axis=1
        )


class SqueezeNet(nn.Layer):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2D(3, 96, kernel_size=7, stride=2),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2D(3, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2D(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(), nn.AdaptiveAvgPool2D((1, 1))
        )

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                if m is final_conv:
                    m.weight.set_value(paddle.randn(m.weight.shape) * 0.01)  # normal_(0, 0.01)
                else:
                    # kaiming_uniform_(fan_in) 等效分布：U(-sqrt(6/fan_in), sqrt(6/fan_in))
                    bound = math.sqrt(6.0 / fan_in)
                    m.weight.set_value(paddle.uniform(m.weight.shape, min=-bound, max=bound))
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros(m.bias.shape))  # constant_(bias, 0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return paddle.flatten(x, start_axis=1)  # start_dim → start_axis
