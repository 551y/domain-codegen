# PyTorch → PaddlePaddle 转换产物（模型部署适配知识库增强生成）
# 对齐说明：
#   - torch.nn.Conv2d → paddle.nn.Conv2D：bias → bias_attr；默认均含 bias
#   - torch.nn.BatchNorm2d → paddle.nn.BatchNorm2D：eps → epsilon、affine → weight_attr/bias_attr、
#     track_running_stats → use_global_stats（本模型均使用默认值，无需显式传参）
#   - torch.nn.init.* → paddle.nn.initializer.* 等效分布：
#     torch 为构造后原地初始化，Paddle 侧在 _init_weights 中以 set_value 按等效分布重写，保持 init_weights 语义
import math
from typing import cast, Dict, List, Union

import paddle
import paddle.nn as nn

__all__ = ["VGG"]


class VGG(nn.Layer):
    def __init__(
        self, features: nn.Layer, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2D((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._init_weights()

    def _init_weights(self) -> None:
        # self.modules() → self.sublayers()：Paddle Layer 的递归子层迭代器
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # kaiming_normal_(nonlinearity="relu", fan_in) 等效分布：N(0, sqrt(2/fan_in))
                fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                m.weight.set_value(paddle.randn(m.weight.shape) * math.sqrt(2.0 / fan_in))
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros(m.bias.shape))  # constant_(bias, 0)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))    # constant_(weight, 1)
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.Linear):
                m.weight.set_value(paddle.randn(m.weight.shape) * 0.01)  # normal_(0, 0.01)
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, start_axis=1)  # start_dim → start_axis
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Layer] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
