import paddle
from functools import partial
from typing import Any, Optional
import paddle.nn as nn
__all__ = ['AlexNet']


class AlexNet(paddle.nn.Layer):

    def __init__(self, num_classes: int=1000, dropout: float=0.5) ->None:
        super().__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle .nn.Conv2D(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D( in_channels=384, out_channels=256, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2)
        )
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(6, 6))
        self.classifier = paddle.nn.Sequential(
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=4096, out_features=4096),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.classifier(x)
        return x
