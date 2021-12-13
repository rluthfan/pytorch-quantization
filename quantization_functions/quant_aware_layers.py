import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class QParam(nn.Module):
    def __init__(self, num_bit, sign=False):
        super(QParam, self).__init__()
        vmin = torch.tensor(float('inf'), requires_grad=False)
        vmax = torch.tensor(float('-inf'), requires_grad=False)
        zero_point = torch.tensor(0., requires_grad=False)
        scale = torch.tensor(0., requires_grad=False)
        sign = torch.tensor(sign)
        num_bit = torch.as_tensor(num_bit)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('scale', scale)
        self.register_buffer('num_bit', num_bit)
        self.register_buffer('vmin', vmin)
        self.register_buffer('vmax', vmax)
        self.register_buffer('sign', sign)
        if not sign:
            qmin, qmax = 0, 2 ** self.num_bit - 1
        else:
            qmin, qmax = - 2 ** (self.num_bit - 1), 2 ** (self.num_bit - 1) - 1
        self.qmin, self.qmax = torch.as_tensor(qmin), torch.as_tensor(qmax)

    def calculate_scale_zero(self):
        self.scale = (self.vmax - self.vmin) / (self.qmax - self.qmin)
        self.zero_point = self.qmax - self.vmax / self.scale
        self.zero_point.round_()

    def quantize_update(self, tensor):
        t_min, t_max = tensor.min(), tensor.max()
        if self.vmin > t_min:
            self.vmin = t_min
        if self.vmax < t_max:
            self.vmax = t_max
        self.calculate_scale_zero()
        return tensor

    def quantize_tensor(self, tensor):
        x = tensor / self.scale + self.zero_point
        x.clamp_(self.qmin, self.qmax)
        x.round_()
        return x

    def dequantize_tensor(self, tensor):
        return (tensor - self.zero_point) * self.scale


def fold_bn(conv_w, conv_b, mean, var, gamma, beta, eps=1e-5):
    ch = var.shape[0]
    gamma_ = gamma / torch.sqrt(var + eps)
    weight = conv_w * gamma_.view(ch, 1, 1, 1)
    if conv_b:
        bias = gamma_ * (conv_b - mean) + beta
    else:
        bias = -gamma_ * mean + beta
    return weight, bias


class CConvBNReLU2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, dilation=1
                 , q_num_bit=8, start=False, affine=True, relu=False, state_dict_names=[]):
        super(CConvBNReLU2d, self).__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.in_channel, self.out_channel, self.kernel_size = in_channel, out_channel, kernel_size
        self.stride, self.padding, self.dilation = stride, padding, dilation,
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1]),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.randn(out_channel), requires_grad=True) if bias else None

        self.affine = affine
        self.bn_weight = nn.Parameter(torch.ones(out_channel), requires_grad=self.affine)
        self.bn_bias = nn.Parameter(torch.zeros(out_channel), requires_grad=self.affine)
        self.running_mean = nn.Parameter(torch.zeros(out_channel), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_channel), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.Tensor(0))

        self.q_w = QParam(q_num_bit)
        self.q_b = QParam(32, sign=True)
        self.q_in = QParam(q_num_bit) if start else None
        self.q_out = QParam(q_num_bit)
        self.if_quantize_forward = False
        self.relu = relu
        self.state_dict_names = state_dict_names
        self.start = start

    def forward(self, x):
        if not self.if_quantize_forward:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
            x = F.batch_norm(x, self.running_mean, self.running_var, self.bn_weight, self.bn_bias)
            if self.relu:
                x = F.relu(x)
            return x
        else:
            if self.start:
                self.q_in.quantize_update(x)
                x = FakeQuantize.apply(x, self.q_in)

            weight, bias = fold_bn(self.weight, self.bias, self.running_mean, self.running_var, self.bn_weight,
                                   self.bn_bias)
            if self.q_w:
                self.q_w.quantize_update(weight)
                weight = FakeQuantize.apply(weight, self.q_w)
            if self.bias:
                self.q_b.scale = self.q_w.scale
                self.q_b.zero_point = torch.tensor(0)
                # self.q_b.quantize_update(bias)
                bias = FakeQuantize.apply(bias, self.q_b)

            stride, padding, dilation = self.stride, self.padding, self.dilation
            x = F.conv2d(x, weight, bias, stride, padding, dilation)

            if self.q_out:
                self.q_out.quantize_update(x)
                x = FakeQuantize.apply(x, self.q_out)
            if self.relu:
                x = F.relu(x)
            return x

    def load_pretrained(self, state_dict):
        assert len(self.state_dict_names) == 7
        layers = [self.weight, self.bias, self.bn_weight, self.bn_bias, self.running_mean, self.running_var,
                  self.num_batches_tracked]
        for name, layer in zip(self.state_dict_names, layers):
            if name is not None and name in state_dict:
                if 'num_batches_tracked' not in name:
                    try:
                        assert layer.data.shape == state_dict[name].shape
                    except Exception as e:
                        print('!!!', name, layer.data.shape, state_dict[name].shape)
                layer.data = state_dict[name]
                # print('loaded', name)
                state_dict.pop(name)

    def quantize(self, if_quantize):
        self.if_quantize_forward = if_quantize


class CLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, q_num_bit=8, pretrained_name=""):
        super(CLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=True) if bias else None
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

        self.q_w = QParam(q_num_bit)
        self.q_b = QParam(32, sign=True) if bias else None
        self.q_out = QParam(q_num_bit)

        self.pretrained_name = pretrained_name
        self.if_quantize_forward = False

    def forward(self, x):
        if not self.if_quantize_forward:
            y = F.linear(x, self.weight, self.bias)
            return y
        else:
            self.q_w.quantize_update(self.weight)
            weight = FakeQuantize.apply(self.weight, self.q_w)

            if self.bias is not None:
                self.q_b.scale = self.q_w.scale
                self.q_b.zero_point = torch.tensor(0)
                # self.q_b.quantize_update(self.bias)
                bias = FakeQuantize.apply(self.bias, self.q_b)
            else:
                bias = None

            y = F.linear(x, weight, bias)
            if self.q_out:
                self.q_out.quantize_update(y)
                y = FakeQuantize.apply(y, self.q_out)
            return y

    def quantize(self, if_quantize):
        self.if_quantize_forward = if_quantize

    def load_pretrained(self, state_dict):
        pre_name = self.pretrained_name + '.' if self.pretrained_name else ""
        for name, data in [(f'fc.weight', self.weight),
                           (f'fc.bias', self.bias)]:
            data_name = pre_name + name
            if data_name in state_dict and data.data.shape == state_dict[data_name].shape:
                data.data = state_dict[data_name]
                state_dict.pop(data_name)
                # print(f'loaded {data_name}')


class CAdd(nn.Module):
    def __init__(self, q_num_bit=8):
        super(CAdd, self).__init__()
        self.q_out = QParam(q_num_bit)
        self.if_quantize_forward = False

    def forward(self, *xs):
        if not self.if_quantize_forward:
            return sum(xs)
        else:
            y = sum(xs)
            self.q_out.quantize_update(y)
            y = FakeQuantize.apply(y, self.q_out)
            return y

    def quantize(self, if_quantize):
        self.if_quantize_forward = if_quantize
