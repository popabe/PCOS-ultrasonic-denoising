from se_attention import SEAttention

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.se = SEAttention(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        return out + x  # 残差连接
