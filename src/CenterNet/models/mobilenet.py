import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet
from torchvision.models import MobileNetV2 as _MobileNetV2
from torchvision.models.mobilenet import ConvBNReLU
from .keypoint_detector import KeypointDetector


class MobileNetV2Backbone(_MobileNetV2):
    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        outs = []
        for f in self.features:
            y = f(x)
            if x.shape[2] != y.shape[2]:
                outs.append(x)
            x = y
        return outs


class MobileNetV2(KeypointDetector):
    stride = 4

    def __init__(self, backbone, num_classes):
        super().__init__()

        backbone.__class__ = MobileNetV2Backbone
        del backbone.classifier
        self.backbone = backbone

        self.layer3 = ConvBNReLU(96, 32, 3, 1)
        self.layer2 = ConvBNReLU(32 + 32, 24, 3, 1)
        self.layer1 = ConvBNReLU(24 + 24, 64, 3, 1)
        self.fc = nn.Conv2d(64, num_classes, 1, 1, 0)

    def _forward(self, x):
        outs = self.backbone(x)
        l1, l2, l3 = outs[2:]  # stride = 4

        x = self.layer3(l3)
        x = F.interpolate(x, l2.shape[2:])

        x = torch.cat((x, l2), dim=1)
        x = self.layer2(x)
        x = F.interpolate(x, l1.shape[2:])

        x = torch.cat((x, l1), dim=1)
        x = self.layer1(x)

        x = self.fc(x)

        return x
