import paddle
from functools import partial
from typing import Any, Optional
import paddle.nn as nn
import paddle.nn.initializer as init
__all__ = ['SqueezeNet']


class Fire(paddle.nn.Layer):

    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes:
        int, expand3x3_planes: int) ->None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = paddle.nn.Conv2D(in_channels=inplanes, out_channels=
            squeeze_planes, kernel_size=1)
        self.squeeze_activation = paddle.nn.ReLU()
        self.expand1x1 = paddle.nn.Conv2D(in_channels=squeeze_planes,
            out_channels=expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = paddle.nn.ReLU()
        self.expand3x3 = paddle.nn.Conv2D(in_channels=squeeze_planes,
            out_channels=expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = paddle.nn.ReLU()

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return paddle.concat(x=[self.expand1x1_activation(self.expand1x1(x)
            ), self.expand3x3_activation(self.expand3x3(x))], axis=1)


class SqueezeNet(paddle.nn.Layer):

    def __init__(self, version: str='1_0', num_classes: int=1000, dropout:
        float=0.5) ->None:
        super().__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = paddle.nn.Sequential(
                paddle.nn.Conv2D( in_channels=3, out_channels=96, kernel_size=7, stride=2),
                paddle.nn.ReLU(),
                paddle.nn.MaxPool2D(kernel_size=3, stride =2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                paddle.nn.MaxPool2D( kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                paddle.nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256)
            )
        elif version == '1_1':
            self.features = paddle.nn.Sequential(
                paddle.nn.Conv2D( in_channels=3, out_channels=64, kernel_size=3, stride=2),
                paddle.nn.ReLU(),
                paddle.nn.MaxPool2D(kernel_size=3, stride =2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                paddle.nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode =True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                paddle.nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True ),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire( 384, 64, 256, 256),
                Fire(512, 64, 256, 256)
            )
        else:
            raise ValueError(
                f'Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected'
                )
        final_conv = paddle.nn.Conv2D(in_channels=512, out_channels=self.
            num_classes, kernel_size=1)
        self.classifier = paddle.nn.Sequential(
            paddle.nn.Dropout(p=dropout),
            final_conv,
            paddle.nn.ReLU(),
            paddle.nn.AdaptiveAvgPool2D( output_size=(1, 1)
        ))
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                if m is final_conv:
                    tmp_initializer = paddle.nn.initializer.Normal(mean=0.0,
                        std=0.01)
                    m.weight.set_value(paddle.create_parameter(shape=m.
                        weight.shape, dtype=m.weight.dtype,
                        default_initializer=tmp_initializer))
                else:
                    tmp_initializer = paddle.nn.initializer.KaimingUniform()
                    m.weight.set_value(paddle.create_parameter(shape=m.
                        weight.shape, dtype=m.weight.dtype,
                        default_initializer=tmp_initializer))
                if m.bias is not None:
                    tmp_initializer = paddle.nn.initializer.Constant(value=0)
                    m.bias.set_value(paddle.create_parameter(shape=m.bias.
                        shape, dtype=m.bias.dtype, default_initializer=
                        tmp_initializer))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return paddle.flatten(x=x, start_axis=1)
