"""
DroneNet —— 无人机轻量视觉识别骨干（PyTorch）

面向机载算力受限场景的目标分类网络：深度可分离卷积 + SE 通道注意力，
尾部含双线性上采样特征融合分支。可作为 PyTorch → ONNX 导出适配的输入样例。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力：全局池化压缩 + 两 FC 生成通道权重。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.view(b, c, 1, 1)


class DWConvBlock(nn.Module):
    """深度可分离卷积块：depthwise 3x3 + pointwise 1x1 + BN + ReLU。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return self.se(x)


class DroneNet(nn.Module):
    """轻量分类骨干：stem + 3 级 DWConvBlock + 上采样融合 + 分类头。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = DWConvBlock(16, 32, stride=2)
        self.stage2 = DWConvBlock(32, 64, stride=2)
        self.stage3 = DWConvBlock(64, 128, stride=2)
        self.lateral = nn.Conv2d(64, 128, 1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # 1/2
        x = self.stage1(x)        # 1/4
        mid = self.stage2(x)      # 1/8
        high = self.stage3(mid)   # 1/16
        # 上采样融合：高层语义双线性插值回 1/8，与中层侧向投影相加
        up = F.interpolate(high, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        fused = up + self.lateral(mid)
        out = self.pool(fused).flatten(1)
        return self.fc(out)


if __name__ == "__main__":
    model = DroneNet(num_classes=10).eval()
    dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(dummy)
    print("output:", y.shape)
