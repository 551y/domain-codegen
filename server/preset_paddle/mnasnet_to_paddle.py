import paddle
import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import paddle.nn as nn
from paddle import Tensor
__all__ = ['MNASNet']
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(paddle.nn.Layer):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride:
        int, expansion_factor: int, bn_momentum: float=0.1) ->None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')
        if kernel_size not in [3, 5]:
            raise ValueError(
                f'kernel_size should be 3 or 5 instead of {kernel_size}')
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels= in_ch, out_channels=mid_ch, kernel_size=1, bias_attr=False),
            paddle.nn.BatchNorm2D(num_features=mid_ch, momentum=1.0 - bn_momentum),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels= mid_ch, out_channels=mid_ch, kernel_size=kernel_size, padding= kernel_size // 2, stride=stride, groups=mid_ch, bias_attr=False ),
            paddle.nn.BatchNorm2D(num_features=mid_ch, momentum=1.0 - bn_momentum),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels= mid_ch, out_channels=out_ch, kernel_size=1, bias_attr=False),
            paddle.nn.BatchNorm2D(num_features=out_ch, momentum=1.0 - bn_momentum)
        )

    def forward(self, input: paddle.Tensor) ->paddle.Tensor:
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch: int, out_ch: int, kernel_size: int, stride: int,
    exp_factor: int, repeats: int, bn_momentum: float) ->paddle.nn.Sequential:
    """Creates a stack of inverted residuals."""
    if repeats < 1:
        raise ValueError(f'repeats should be >= 1, instead got {repeats}')
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride,
        exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1,
            exp_factor, bn_momentum=bn_momentum))
    return paddle.nn.Sequential(
        first,
        *remaining
    )


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float=0.9
    ) ->int:
    """Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88."""
    if not 0.0 < round_up_bias < 1.0:
        raise ValueError(
            f'round_up_bias should be greater than 0.0 and smaller than 1.0 instead of {round_up_bias}'
            )
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) ->List[int]:
    """Scales tensor depths as in reference MobileNet code, prefers rounding up
    rather than down."""
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(paddle.nn.Layer):
    """MNASNet, as described in https://arxiv.org/abs/1807.11626. This
    implements the B1 variant of the model.
    1000
    """
    _version = 2

    def __init__(self, alpha: float, num_classes: int=1000, dropout: float=0.2
        ) ->None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(
                f'alpha should be greater than 0.0 instead of {alpha}')
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [paddle.nn.Conv2D(in_channels=3, out_channels=depths[0],
            kernel_size=3, padding=1, stride=2, bias_attr=False), paddle.nn
            .BatchNorm2D(num_features=depths[0], momentum=1.0 -
            _BN_MOMENTUM), paddle.nn.ReLU(), paddle.nn.Conv2D(in_channels=
            depths[0], out_channels=depths[0], kernel_size=3, padding=1,
            stride=1, groups=depths[0], bias_attr=False), paddle.nn.
            BatchNorm2D(num_features=depths[0], momentum=1.0 - _BN_MOMENTUM
            ), paddle.nn.ReLU(), paddle.nn.Conv2D(in_channels=depths[0],
            out_channels=depths[1], kernel_size=1, padding=0, stride=1,
            bias_attr=False), paddle.nn.BatchNorm2D(num_features=depths[1],
            momentum=1.0 - _BN_MOMENTUM), _stack(depths[1], depths[2], 3, 2,
            3, 3, _BN_MOMENTUM), _stack(depths[2], depths[3], 5, 2, 3, 3,
            _BN_MOMENTUM), _stack(depths[3], depths[4], 5, 2, 6, 3,
            _BN_MOMENTUM), _stack(depths[4], depths[5], 3, 1, 6, 2,
            _BN_MOMENTUM), _stack(depths[5], depths[6], 5, 2, 6, 4,
            _BN_MOMENTUM), _stack(depths[6], depths[7], 3, 1, 6, 1,
            _BN_MOMENTUM), paddle.nn.Conv2D(in_channels=depths[7],
            out_channels=1280, kernel_size=1, padding=0, stride=1,
            bias_attr=False), paddle.nn.BatchNorm2D(num_features=1280,
            momentum=1.0 - _BN_MOMENTUM), paddle.nn.ReLU()]
        self.layers = paddle.nn.Sequential(
            *layers
        )
        self.classifier = paddle.nn.Sequential(
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=1280, out_features=num_classes)
        )
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                tmp_initializer = paddle.nn.initializer.KaimingNormal(
                    nonlinearity='relu')
                m.weight.set_value(paddle.create_parameter(shape=m.weight.
                    shape, dtype=m.weight.dtype, default_initializer=
                    tmp_initializer))
                if m.bias is not None:
                    tmp_initializer = paddle.nn.initializer.Constant(value=0)
                    m.bias.set_value(paddle.create_parameter(shape=m.bias.
                        shape, dtype=m.bias.dtype, default_initializer=
                        tmp_initializer))
            elif isinstance(m, paddle.nn.BatchNorm2D):
                tmp_initializer = paddle.nn.initializer.Constant(value=1)
                m.weight.set_value(paddle.create_parameter(shape=m.weight.
                    shape, dtype=m.weight.dtype, default_initializer=
                    tmp_initializer))
                tmp_initializer = paddle.nn.initializer.Constant(value=0)
                m.bias.set_value(paddle.create_parameter(shape=m.bias.shape,
                    dtype=m.bias.dtype, default_initializer=tmp_initializer))
            elif isinstance(m, paddle.nn.Linear):
                tmp_initializer = paddle.nn.initializer.KaimingUniform(
                    nonlinearity='sigmoid')
                m.weight.set_value(paddle.create_parameter(shape=m.weight.
                    shape, dtype=m.weight.dtype, default_initializer=
                    tmp_initializer))
                tmp_initializer = paddle.nn.initializer.Constant(value=0)
                m.bias.set_value(paddle.create_parameter(shape=m.bias.shape,
                    dtype=m.bias.dtype, default_initializer=tmp_initializer))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.layers(x)
        x = x.mean(axis=[2, 3])
        return self.classifier(x)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str,
        local_metadata: Dict, strict: bool, missing_keys: List[str],
        unexpected_keys: List[str], error_msgs: List[str]) ->None:
        version = local_metadata.get('version', None)
        if version not in [1, 2]:
            raise ValueError(
                f'version shluld be set to 1 or 2 instead of {version}')
        if version == 1 and not self.alpha == 1.0:
            depths = _get_depths(self.alpha)
            v1_stem = [paddle.nn.Conv2D(in_channels=3, out_channels=32,
                kernel_size=3, padding=1, stride=2, bias_attr=False),
                paddle.nn.BatchNorm2D(num_features=32, momentum=1.0 -
                _BN_MOMENTUM), paddle.nn.ReLU(), paddle.nn.Conv2D(
                in_channels=32, out_channels=32, kernel_size=3, padding=1,
                stride=1, groups=32, bias_attr=False), paddle.nn.
                BatchNorm2D(num_features=32, momentum=1.0 - _BN_MOMENTUM),
                paddle.nn.ReLU(), paddle.nn.Conv2D(in_channels=32,
                out_channels=16, kernel_size=1, padding=0, stride=1,
                bias_attr=False), paddle.nn.BatchNorm2D(num_features=16,
                momentum=1.0 - _BN_MOMENTUM), _stack(16, depths[2], 3, 2, 3,
                3, _BN_MOMENTUM)]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer
            self._version = 1
            warnings.warn(
                'A new version of MNASNet model has been implemented. Your checkpoint was saved using the previous version. This checkpoint will load and work as before, but you may want to upgrade by training a newer model or transfer learning from an updated ImageNet checkpoint.'
                , UserWarning)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)
