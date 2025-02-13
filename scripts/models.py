import torch
import torch.nn as nn


class ResidualLearningNet(nn.Module):
    def __init__(self, channels=14):
        super(ResidualLearningNet, self).__init__()
        self.residual_layer = self.make_layer(10)

        self.input = nn.Conv2d(in_channels=channels, out_channels=64,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=channels,
                                kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.relu(self.input(x2))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ResidualLearningNet con Depthwise Separable Convolutions
class LightLearningNet(nn.Module):
    def __init__(self, channels=14):
        super(LightLearningNet, self).__init__()
        self.residual_layer = self.make_layer(10)  # Ridotto a 6 layer per efficienza
        self.input = DepthwiseSeparableConv(in_channels=channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.relu(self.input(x2))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, mlp_ratio=2.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape to (B, H*W, C) for MHSA
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output  # Residual connection

        x = self.norm2(x)
        x = x + self.mlp(x)  # Residual connection with FFN

        x = x.permute(0, 2, 1).view(B, C, H, W)  # Back to (B, C, H, W)
        return x


class Tnet(nn.Module):
    def __init__(self, channels=14, hidden_dim=16, num_transformers=2):
        super(Tnet, self).__init__()
        self.input = DepthwiseSeparableConv(in_channels=channels, out_channels=hidden_dim, kernel_size=3, stride=1,
                                            padding=1)
        self.residual_layer = self.make_layer(10, hidden_dim)  # Ridotto a 6 layer per efficienza
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(hidden_dim) for _ in range(num_transformers)])
        self.output = nn.Conv2d(in_channels=hidden_dim, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def make_layer(self, count, hidden_dim):
        layers = []
        for _ in range(count):
            layers.append(
                DepthwiseSeparableConv(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1,
                                       padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.transformer_blocks(out)  # Aggiunto il Transformer Block
        out = self.output(out)
        out = torch.add(out, residual)  # Residual connection
        return out

