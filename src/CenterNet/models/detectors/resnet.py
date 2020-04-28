import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet as _ResNet
from torchvision.models.resnet import BasicBlock
from .detector import GeneralDetector


class ResNetDetector(GeneralDetector):
    stride = 4

    def _make_layer(self, inplanes, planes, blocks):
        _norm_layer = nn.BatchNorm2d
        _block = BasicBlock

        downsample = None
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False), _norm_layer(planes),
            )

        layer = nn.Sequential(
            _block(inplanes, planes, downsample=downsample, norm_layer=_norm_layer),
            _block(planes, planes, norm_layer=_norm_layer),
        )

        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return layer
