# PyTorch → PaddlePaddle 转换产物（模型部署适配知识库增强生成）
# 对齐说明：
#   - torch.chunk(x, 2, dim=1) → x.chunk(2, axis=1)：参数变换 dim → axis
#   - torch.cat → paddle.concat（dim → axis）；x.size() → x.shape；view → reshape
#   - torch.transpose(x, 1, 2) → paddle.transpose(x, perm=[...])：语义对齐——torch 交换两个维度，
#     Paddle 需给出完整维度排列；contiguous() 是 torch 内存布局概念，Paddle 框架内部处理，无需对应调用
#   - nn.Conv2d(bias=False) → nn.Conv2D(bias_attr=False)：参数变换 bias → bias_attr
#   - x.mean([2, 3]) → x.mean(axis=[2, 3])
from typing import Callable, List

import paddle
import paddle.nn as nn

__all__ = ["ShuffleNetV2"]


def channel_shuffle(x: paddle.Tensor, groups: int) -> paddle.Tensor:
    # 语义对齐：Paddle 有等价直连算子 paddle.nn.ChannelShuffle，此处保留 reshape/transpose
    # 的逐算子改写路径，与源实现逐行对应
    batchsize, num_channels, height, width = x.shape  # x.size() → x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape([batchsize, groups, channels_per_group, height, width])

    # torch.transpose(x, 1, 2) 交换 dims 1/2 → paddle.transpose 完整 perm；contiguous() 无需对应
    x = paddle.transpose(x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = x.reshape([batchsize, num_channels, height, width])

    return x


class InvertedResidual(nn.Layer):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2D(inp),
                nn.Conv2D(inp, branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(branch_features),
                nn.ReLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2D(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,  # 参数变换：bias=False → bias_attr=False
            ),
            nn.BatchNorm2D(branch_features),
            nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2D(branch_features),
            nn.Conv2D(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(branch_features),
            nn.ReLU(),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2D:
        # groups 语义一致；bias → bias_attr（False 表示关闭偏置）
        return nn.Conv2D(i, o, kernel_size, stride, padding, groups=i, bias_attr=None if bias else False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, axis=1)  # torch.chunk(dim=) → axis=
            out = paddle.concat((x1, self.branch2(x2)), axis=1)
        else:
            out = paddle.concat((self.branch1(x), self.branch2(x)), axis=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Layer):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nn.Layer] = InvertedResidual,
    ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2D(input_channels, output_channels, 3, 2, 1, bias_attr=False),
            nn.BatchNorm2D(output_channels),
            nn.ReLU(),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2D(input_channels, output_channels, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(output_channels),
            nn.ReLU(),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean(axis=[2, 3])  # globalpool：torch mean([2,3]) → mean(axis=[2,3])
        x = self.fc(x)
        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self._forward_impl(x)
