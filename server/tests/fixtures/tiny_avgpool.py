import torch.nn as nn

class TinyAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)
