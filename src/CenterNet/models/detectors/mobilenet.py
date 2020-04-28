import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenet import InvertedResidual
from .detector import GeneralDetector


class MobileNetV2Detector(GeneralDetector):
    stride = 4

    def _make_layer(self, in_planes, out_planes, layer_idx):
        layer = nn.Sequential(
            InvertedResidual(in_planes, out_planes, 1, 6),
            InvertedResidual(out_planes, out_planes, 1, 6),
        )

        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return layer
