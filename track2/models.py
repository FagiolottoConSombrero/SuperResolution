import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


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


class SecondLightResidualNet(nn.Module):
    def __init__(self, channels=14):
        super(SecondLightResidualNet, self).__init__()
        self.residual_layer = self.make_layer(7)

        self.input = nn.Conv2d(in_channels=channels + 3, out_channels=64,
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
            layers.append(DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x2, y2):
        residual = x2
        out = self.relu(self.input(torch.cat((x2, y2), 1)))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


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


class SecondResidualNet(nn.Module):
    def __init__(self, channels=14):
        super(SecondResidualNet, self).__init__()
        self.residual_layer = self.make_layer(7)

        self.input = nn.Conv2d(in_channels=channels + 3, out_channels=64,
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

    def forward(self, x2, y2):
        residual = x2
        out = self.relu(self.input(torch.cat((x2, y2), 1)))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
                   out_channels,
                   kernel_size,
                   bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    """
    Modules will be added to the Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0,
                      bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s,
                                   bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(
            1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

class SPAB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att

class SPAN_2(nn.Module):
        """
        Swift Parameter-free Attention Network for Efficient Super-Resolution
        """

        def __init__(self,
                     num_in_ch,
                     num_out_ch,
                     feature_channels=48,
                     bias=True,
                     ):
            super(SPAN_2, self).__init__()

            in_channels = num_in_ch
            out_channels = num_out_ch

            self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
            self.block_1 = SPAB(feature_channels, bias=bias)
            self.block_2 = SPAB(feature_channels, bias=bias)

            self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
            self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

            self.upsampler = conv_layer(feature_channels, out_channels, kernel_size=1, bias=True)

        def forward(self, x, y):
            out_feature = self.conv_1((torch.cat((x, y), 1)))

            out_b1, _, att1 = self.block_1(out_feature)
            out_b2, out_b1_2, att2 = self.block_2(out_b1)

            out_b2 = self.conv_2(out_b2)
            out = self.conv_cat(torch.cat([out_feature, out_b2, out_b1, out_b1_2], 1))
            output = self.upsampler(out)

            return output


class SPAN(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=48,
                 bias=True,
                 ):
        super(SPAN, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = conv_layer(feature_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1, _, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)

        out_b2 = self.conv_2(out_b2)
        out = self.conv_cat(torch.cat([out_feature, out_b2, out_b1, out_b1_2], 1))
        output = self.upsampler(out)

        return output