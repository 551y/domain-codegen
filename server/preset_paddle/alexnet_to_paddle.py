# PyTorch → PaddlePaddle 转换产物（模型部署适配知识库增强生成）
# 对齐说明：
#   - torch.nn.Conv2d → paddle.nn.Conv2D：bias 参数由 bias_attr 控制（默认均含 bias）
#   - torch.nn.ReLU(inplace=True) → paddle.nn.ReLU()：Paddle 无 inplace 参数（原地优化由框架内部处理），去除后语义一致
#   - torch.flatten(x, 1) → paddle.flatten(x, start_axis=1)：参数名 start_dim → start_axis
import paddle
import paddle.nn as nn

__all__ = ["AlexNet"]


class AlexNet(nn.Layer):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),  # 语义对齐：Paddle ReLU 无 inplace 参数
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, start_axis=1)  # 参数变换：start_dim → start_axis
        x = self.classifier(x)
        return x
