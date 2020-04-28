import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenet import ConvBNReLU
from .detector import GeneralDetector


class FCNDetector(GeneralDetector):
    stride = 4

    def _make_layer(self, in_planes, out_planes, layer_idx):
        layer = nn.Sequential(
            ConvBNReLU(in_planes, out_planes, 3, 1, 1),
            ConvBNReLU(out_planes, out_planes, 3, 1, 1),
        )

        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return layer
