import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_aware_layers import CConvBNReLU2d, CLinear, CAdd


class CBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, q_num_bit, layer_name=""):
        super(CBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        stride = outplanes // inplanes
        downsample = stride == 2
        self.layer_name = layer_name

        state_dict_names1 = [layer_name + '.' + name for name in
                             ['conv1.weight', "", 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                              'bn1.num_batches_tracked']]
        self.conv1 = CConvBNReLU2d(
            inplanes, outplanes, (3, 3), stride, padding=1, bias=False, dilation=1, 
            q_num_bit=q_num_bit, affine=True, relu=True, state_dict_names=state_dict_names1
        )

        state_dict_names2 = [layer_name + '.' + name for name in
                             ['conv2.weight', "", 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
                              'bn2.num_batches_tracked']]
        self.conv2 = CConvBNReLU2d(
            outplanes, outplanes, (3, 3), (1, 1), padding=1, bias=False, dilation=1, 
            q_num_bit=q_num_bit, affine=True, relu=False, state_dict_names=state_dict_names2
        )

        self.act2 = nn.ReLU(inplace=True)
        self.stride = stride

        if downsample:
            state_dict_names_d = [layer_name + '.' + name for name in
                                  ['downsample.0.weight', "", 'downsample.1.weight', 'downsample.1.bias',
                                   'downsample.1.running_mean', 'downsample.1.running_var',
                                   'downsample.1.num_batches_tracked']]
            self.downsample = CConvBNReLU2d(
                inplanes, outplanes, kernel_size=(1, 1), stride=(2, 2), bias=False,
                q_num_bit=q_num_bit, affine=True, relu=False, state_dict_names=state_dict_names_d
            )
        else:
            self.downsample = None
        self.add = CAdd(q_num_bit=q_num_bit)
        self.act2 = nn.ReLU()

    def load_pretrained(self, state_dict):
        self.conv1.load_pretrained(state_dict)
        self.conv2.load_pretrained(state_dict)
        if self.downsample:
            self.downsample.load_pretrained(state_dict)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x = self.add(x, shortcut)
        x = self.act2(x)
        return x

    def quantize(self, if_quantize):
        self.conv1.quantize(if_quantize)
        self.conv2.quantize(if_quantize)
        self.add.quantize(if_quantize)
        if self.downsample:
            self.downsample.quantize(if_quantize)


class CResnet18(nn.Module):
    def __init__(self, num_class, q_num_bit, pretrained=True):
        super(CResnet18, self).__init__()
        state_dict_names = ['conv1.weight', "", 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                            'bn1.num_batches_tracked']
        self.conv1 = CConvBNReLU2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False, start=True,
            q_num_bit=q_num_bit, affine=True, relu=True, state_dict_names=state_dict_names
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(CBasicBlock(64, 64, q_num_bit, layer_name='layer1.0'),
                                    CBasicBlock(64, 64, q_num_bit, layer_name='layer1.1'))
        self.layer2 = nn.Sequential(CBasicBlock(64, 128, q_num_bit, layer_name='layer2.0'),
                                    CBasicBlock(128, 128, q_num_bit, layer_name='layer2.1'))
        self.layer3 = nn.Sequential(CBasicBlock(128, 256, q_num_bit, layer_name='layer3.0'),
                                    CBasicBlock(256, 256, q_num_bit, layer_name='layer3.1'))
        self.layer4 = nn.Sequential(CBasicBlock(256, 512, q_num_bit, layer_name='layer4.0'),
                                    CBasicBlock(512, 512, q_num_bit, layer_name='layer4.1'))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = CLinear(512, num_class, q_num_bit=q_num_bit)
        if pretrained:
            if pretrained is True:
                import timm
                state_dict = timm.create_model('resnet18', pretrained=True).state_dict()
            else:
                with open(pretrained, 'rb') as f:
                    state_dict = torch.load(f)
            self.conv1.load_pretrained(state_dict)
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    block.load_pretrained(state_dict)
            self.fc.load_pretrained(state_dict)
            print('remained state dict', state_dict.keys())

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def quantize(self, if_quantize):
        self.conv1.quantize(if_quantize)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.quantize(if_quantize)
        self.fc.quantize(if_quantize)