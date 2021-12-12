import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_bn(conv_w, conv_b, mean, var, gamma, beta, eps=1e-5):
    ch = var.shape[0]
    gamma_ = gamma / torch.sqrt(var + eps)
    weight = conv_w * gamma_.view(ch, 1, 1, 1)
    if conv_b:
        bias = gamma_ * (conv_b - mean) + beta
    else:
        bias = -gamma_ * mean + beta
    return weight, bias


class QConvBnReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, relu, stride=1, padding=0, dilation=1):
        super(QConvBnReLU, self).__init__()
        self.weight = nn.Parameter(
            torch.zeros(out_channel, in_channel, kernel_size[0], kernel_size[1], dtype=torch.uint8),
            requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_channel, dtype=torch.int32), requires_grad=False)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        M = torch.tensor(0., requires_grad=False)
        x_zero = torch.tensor(0., requires_grad=False)
        w_zero = torch.tensor(0., requires_grad=False)
        y_zero = torch.tensor(0., requires_grad=False)
        self.register_buffer('M', M)
        self.register_buffer('x_zero', x_zero)
        self.register_buffer('w_zero', w_zero)
        self.register_buffer('y_zero', y_zero)
        self.register_buffer('relu', torch.tensor(relu))

    def convert_from(self, cbr, q_in):
        weight, bias = cbr.weight, None
        bn_weight, bn_bias, bn_mean, bn_var = cbr.bn_weight, cbr.bn_bias, cbr.running_mean, cbr.running_var
        weight, bias = fold_bn(weight, bias, bn_mean, bn_var, bn_weight, bn_bias)

        q_w = cbr.q_w
        q_b = cbr.q_b
        q_b.scale = q_w.scale * q_in.scale
        q_b.zero_point = torch.tensor(0)
        q_out = cbr.q_out

        self.M = q_in.scale * q_w.scale / q_out.scale
        self.w_zero = q_w.zero_point
        self.x_zero = q_in.zero_point
        self.y_zero = q_out.zero_point

        weight = q_w.quantize_tensor(weight)
        bias = q_b.quantize_tensor(bias)
        self.weight.data = weight.round_().type(torch.uint8)
        self.bias.data = bias.round_().type(torch.int32)

    def forward(self, x):
        x = x - self.x_zero
        x = F.pad(x, (self.padding,) * 4, 'constant', 0)
        w = self.weight.float() - self.w_zero
        b = self.bias.float()
        # print(x.dtype,w.dtype,b.dtype)
        y = F.conv2d(x, w, b, self.stride, 0, self.dilation) * self.M + self.y_zero
        if self.relu:
            y[y < self.y_zero] = self.y_zero
        return y.round_()


class QReLU(nn.Module):
    def __init__(self):
        super(QReLU, self).__init__()
        zero_point = torch.tensor(0., requires_grad=False)
        self.register_buffer('zero_point', zero_point)

    def convert_from(self, q_in):
        self.zero_point = q_in.zero_point

    def forward(self, x):
        x[x < self.zero_point] = self.zero_point
        return x.round_()


class QMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(QMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding).round_()


class QAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(QAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, self.output_size)
        return y.round_()


class QLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.uint8), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.int32), requires_grad=False)
        M = torch.tensor(0., requires_grad=False)
        x_zero = torch.tensor(0., requires_grad=False)
        w_zero = torch.tensor(0., requires_grad=False)
        y_zero = torch.tensor(0., requires_grad=False)
        self.register_buffer('M', M)
        self.register_buffer('x_zero', x_zero)
        self.register_buffer('w_zero', w_zero)
        self.register_buffer('y_zero', y_zero)

    def convert_from(self, linear, q_in):
        q_w = linear.q_w
        q_b = linear.q_b
        q_b.scale = q_w.scale * q_in.scale
        q_out = linear.q_out

        self.weight.data = q_w.quantize_tensor(linear.weight).type(torch.uint8)
        self.bias.data = q_b.quantize_tensor(linear.bias).type(torch.int32)
        self.M = q_in.scale * q_w.scale / q_out.scale
        self.w_zero = q_w.zero_point
        self.x_zero = q_in.zero_point
        self.y_zero = q_out.zero_point

    def forward(self, x):
        x = x - self.x_zero
        w = self.weight.float() - self.w_zero
        b = self.bias.float()
        y = F.linear(x, w, b) * self.M + self.y_zero
        return y.round_()


class QAdd(nn.Module):
    def __init__(self):
        super(QAdd, self).__init__()
        x_scales = torch.tensor([1., 1.], requires_grad=False)
        x_zeros = torch.tensor([0., 0.], requires_grad=False)
        self.register_buffer('x_scales', x_scales)
        self.register_buffer('x_zeros', x_zeros)
        y_scale = torch.tensor(0., requires_grad=False)
        y_zero = torch.tensor(0., requires_grad=False)
        self.register_buffer('y_scale', y_scale)
        self.register_buffer('y_zero', y_zero)

    def convert_from(self, add, q_ins):
        scales = torch.tensor([q_in.scale for q_in in q_ins])
        zeros = torch.tensor([q_in.zero_point for q_in in q_ins])
        self.x_scales = scales
        self.x_zeros = zeros
        self.y_scale = add.q_out.scale
        self.y_zero = add.q_out.zero_point

    def forward(self, *xs):
        xs = [(x - zero) * scale for i, (x, scale, zero) in enumerate(zip(xs, self.x_scales, self.x_zeros))]
        return (sum(xs) / self.y_scale + self.y_zero).round_()