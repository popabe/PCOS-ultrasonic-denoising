import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention Module for channel-wise attention.
    可用于U-Net中的任意残差块或卷积块，替换Self-Attention模块。
    """
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),     # 可用ReLU
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
