import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_aware_layers import CConvBNReLU2d, CLinear, CAdd


class CBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, q_num_bit, layer_name="", qat=False):
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
            q_num_bit=q_num_bit, affine=True, relu=True, state_dict_names=state_dict_names1, qat=qat
        )

        state_dict_names2 = [layer_name + '.' + name for name in
                             ['conv2.weight', "", 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
                              'bn2.num_batches_tracked']]
        self.conv2 = CConvBNReLU2d(
            outplanes, outplanes, (3, 3), (1, 1), padding=1, bias=False, dilation=1, 
            q_num_bit=q_num_bit, affine=True, relu=False, state_dict_names=state_dict_names2, qat=qat
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
                q_num_bit=q_num_bit, affine=True, relu=False, state_dict_names=state_dict_names_d, qat=qat
            )
        else:
            self.downsample = None
        self.add = CAdd(q_num_bit=q_num_bit, qat=qat)
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

class CBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, q_num_bit, layer_name="", qat=False):
        super(CBottleneckBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        stride = outplanes*4 // inplanes
        downsample = stride >= 2
        if stride >2: stride=1
        
        self.layer_name = layer_name

        state_dict_names1 = [layer_name + '.' + name for name in
                             ['conv1.weight', "", 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                              'bn1.num_batches_tracked']]
        self.conv1 = CConvBNReLU2d(
            inplanes, outplanes, (1, 1), stride, padding=0, bias=False, dilation=1, 
            q_num_bit=q_num_bit, affine=True, relu=True, state_dict_names=state_dict_names1, qat=qat
        )

        state_dict_names2 = [layer_name + '.' + name for name in
                             ['conv2.weight', "", 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var',
                              'bn2.num_batches_tracked']]
        self.conv2 = CConvBNReLU2d(
            outplanes, outplanes, (3, 3), (1, 1), padding=1, bias=False, dilation=1, 
            q_num_bit=q_num_bit, affine=True, relu=True, state_dict_names=state_dict_names2, qat=qat
        )
        
        state_dict_names3 = [layer_name + '.' + name for name in
                             ['conv3.weight', "", 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var',
                              'bn3.num_batches_tracked']]
        self.conv3 = CConvBNReLU2d(
            outplanes, outplanes*4, (1, 1), (1, 1), padding=0, bias=False, dilation=1, 
            q_num_bit=q_num_bit, affine=True, relu=False, state_dict_names=state_dict_names3, qat=qat
        )

        self.act2 = nn.ReLU(inplace=True)
        self.stride = stride

        if downsample:
            state_dict_names_d = [layer_name + '.' + name for name in
                                  ['downsample.0.weight', "", 'downsample.1.weight', 'downsample.1.bias',
                                   'downsample.1.running_mean', 'downsample.1.running_var',
                                   'downsample.1.num_batches_tracked']]
            self.downsample = CConvBNReLU2d(
                inplanes, outplanes*4, kernel_size=(1, 1), stride=stride, bias=False,
                q_num_bit=q_num_bit, affine=True, relu=False, state_dict_names=state_dict_names_d, qat=qat
            )
        else:
            self.downsample = None
        self.add = CAdd(q_num_bit=q_num_bit, qat=qat)
        self.act2 = nn.ReLU()

    def load_pretrained(self, state_dict):
        self.conv1.load_pretrained(state_dict)
        self.conv2.load_pretrained(state_dict)
        self.conv3.load_pretrained(state_dict)
        if self.downsample:
            self.downsample.load_pretrained(state_dict)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x = self.add(x, shortcut)
        x = self.act2(x)
        return x

    def quantize(self, if_quantize):
        self.conv1.quantize(if_quantize)
        self.conv2.quantize(if_quantize)
        self.conv3.quantize(if_quantize)
        self.add.quantize(if_quantize)
        if self.downsample:
            self.downsample.quantize(if_quantize)


class CResnet(nn.Module):
    def __init__(self, block_type, layers, num_class, q_num_bit, qat=False):
        super(CResnet, self).__init__()
        self.inplanes = 64
        state_dict_names = ['conv1.weight', "", 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
                            'bn1.num_batches_tracked']
        self.conv1 = CConvBNReLU2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False, start=True,
            q_num_bit=q_num_bit, affine=True, relu=True, state_dict_names=state_dict_names, qat=qat
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, 64, layers[0], q_num_bit, "layer1", qat=qat)
        self.layer2 = self._make_layer(block_type, 128, layers[1], q_num_bit, "layer2", qat=qat)
        self.layer3 = self._make_layer(block_type, 256, layers[2], q_num_bit, "layer3", qat=qat)
        self.layer4 = self._make_layer(block_type, 512, layers[3], q_num_bit, "layer4", qat=qat)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = CLinear(512*block_type.expansion, num_class, q_num_bit=q_num_bit, qat=qat)

    def _make_layer(self, block, planes, num_blocks, q_num_bit, layer_name, qat=False):
        layers = []
        layers.append(block(self.inplanes, planes, q_num_bit, layer_name=f"{layer_name}.0", qat=qat))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, q_num_bit, layer_name=f"{layer_name}.{i}", qat=qat))

        return nn.Sequential(*layers)

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

def _resnet_builder(num_class, q_num_bit, block_type, layers, pretrained=True, name='resnet18', qat=False):
    
    model = CResnet(block_type, layers, num_class, q_num_bit, qat=qat)

    if pretrained:
        if pretrained is True:
            import timm
            state_dict = timm.create_model(name, pretrained=True).state_dict()
        else:
            with open(pretrained, 'rb') as f:
                state_dict = torch.load(f)
        model.conv1.load_pretrained(state_dict)
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                block.load_pretrained(state_dict)
        model.fc.load_pretrained(state_dict)
        print('remained state dict', state_dict.keys())
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return model

def CResnet18(num_class, q_num_bit, pretrained=True, qat=False):

    return _resnet_builder(num_class, q_num_bit, CBasicBlock, [2,2,2,2], pretrained, 'resnet18', qat=qat)

def CResnet34(num_class, q_num_bit, pretrained=True, qat=False):

    return _resnet_builder(num_class, q_num_bit, CBasicBlock, [3,4,6,3], pretrained, 'resnet34', qat=qat)

def CResnet50(num_class, q_num_bit, pretrained=True, qat=False):

    return _resnet_builder(num_class, q_num_bit, CBottleneckBlock, [3,4,6,3], pretrained, 'resnet50', qat=qat)

def CResnet101(num_class, q_num_bit, pretrained=True, qat=False):

    return _resnet_builder(num_class, q_num_bit, CBottleneckBlock, [3,4,23,3], pretrained, 'resnet101', qat=qat)

def CResnet152(num_class, q_num_bit, pretrained=True, qat=False):

    return _resnet_builder(num_class, q_num_bit, CBottleneckBlock, [3,8,36,3], pretrained, 'resnet152', qat=qat)
