import torch
import torch.nn as nn

EPS = 1e-3


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
        self.input = DepthwiseSeparableConv(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.relu(self.input(x2))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out