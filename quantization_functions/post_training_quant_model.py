import torch
import torch.nn as nn
import torch.nn.functional as F
from .post_training_quant_layers import QConvBnReLU, QLinear, QAdd, QReLU, QMaxPool2d, QAdaptiveAvgPool2d
from .quant_aware_layers import QParam


class QBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes):
        super(QBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        stride = outplanes // inplanes
        downsample = stride == 2
        self.conv1 = QConvBnReLU(inplanes, outplanes, (3, 3), relu=True, stride=stride, padding=1, dilation=1, )
        self.conv2 = QConvBnReLU(outplanes, outplanes, (3, 3), relu=False, stride=(1, 1), padding=1, dilation=1)
        self.act2 = QReLU()
        self.stride = stride
        if downsample:
            self.downsample = QConvBnReLU(inplanes, outplanes, kernel_size=(1, 1), stride=(2, 2), relu=False)
        else:
            self.downsample = None
        self.add = QAdd()
        self.act2 = QReLU()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x = self.add(x, shortcut)
        x = self.act2(x)
        return x

    def convert_from(self, c_basicblock, q_in):
        self.conv1.convert_from(c_basicblock.conv1, q_in)
        self.conv2.convert_from(c_basicblock.conv2, c_basicblock.conv1.q_out)
        if self.downsample:
            self.downsample.convert_from(c_basicblock.downsample, q_in)
            self.add.convert_from(c_basicblock.add, [c_basicblock.conv2.q_out, c_basicblock.downsample.q_out])
        else:
            self.add.convert_from(c_basicblock.add, [c_basicblock.conv2.q_out, q_in])
        self.act2.convert_from(c_basicblock.add.q_out)

class QBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes):
        super(QBottleneckBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        stride = outplanes*4 // inplanes
        downsample = stride >= 2
        if stride >2: stride=1
        
        self.conv1 = QConvBnReLU(inplanes, outplanes, (1, 1), relu=True, stride=stride, padding=0, dilation=1)
        self.conv2 = QConvBnReLU(outplanes, outplanes, (3, 3), relu=True, stride=(1, 1), padding=1, dilation=1)
        self.conv3 = QConvBnReLU(outplanes, outplanes*4, (1, 1), relu=False, stride=(1, 1), padding=0, dilation=1)
        self.act2 = QReLU()
        self.stride = stride
        if downsample:
            self.downsample = QConvBnReLU(inplanes, outplanes*4, kernel_size=(1, 1), stride=stride, relu=False)
        else:
            self.downsample = None
        self.add = QAdd()
        self.act2 = QReLU()

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

    def convert_from(self, c_bottleneckblock, q_in):
        self.conv1.convert_from(c_bottleneckblock.conv1, q_in)
        self.conv2.convert_from(c_bottleneckblock.conv2, c_bottleneckblock.conv1.q_out)
        self.conv3.convert_from(c_bottleneckblock.conv3, c_bottleneckblock.conv2.q_out)
        if self.downsample:
            self.downsample.convert_from(c_bottleneckblock.downsample, q_in)
            self.add.convert_from(c_bottleneckblock.add, [c_bottleneckblock.conv3.q_out, c_bottleneckblock.downsample.q_out])
        else:
            self.add.convert_from(c_bottleneckblock.add, [c_bottleneckblock.conv3.q_out, q_in])
        self.act2.convert_from(c_bottleneckblock.add.q_out)

class QResnet(nn.Module):
    def __init__(self, block_type, layers, num_class):
        super(QResnet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = QConvBnReLU(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, relu=True)
        self.maxpool = QMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, 64, layers[0])
        self.layer2 = self._make_layer(block_type, 128, layers[1])
        self.layer3 = self._make_layer(block_type, 256, layers[2])
        self.layer4 = self._make_layer(block_type, 512, layers[3])

        self.global_pool = QAdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc = QLinear(512*block_type.expansion, num_class)
        self.q_in = QParam(8)
        self.q_out = QParam(8)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

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
        x = self.q_in.quantize_tensor(x)
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.q_out.dequantize_tensor(x)
        return x

    def convert_from(self, c_resnet):
        self.q_in = c_resnet.conv1.q_in
        self.conv1.convert_from(c_resnet.conv1, c_resnet.conv1.q_in)
        last_q_out = c_resnet.conv1.q_out
        for layer, c_layer in zip([self.layer1, self.layer2, self.layer3, self.layer4],
                                  [c_resnet.layer1, c_resnet.layer2, c_resnet.layer3, c_resnet.layer4]):
            for block, c_block in zip(layer, c_layer):
                block.convert_from(c_block, last_q_out)
                last_q_out = c_block.add.q_out
        self.fc.convert_from(c_resnet.fc, last_q_out)
        self.q_out = c_resnet.fc.q_out

def QResnet18(num_class):
    return QResnet(QBasicBlock, [2,2,2,2], num_class)

def QResnet50(num_class):
    return QResnet(QBottleneckBlock, [3,4,6,3], num_class)
