import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MNASNet as _MNASNet
from torchvision.models.mobilenet import ConvBNReLU
from torchvision.models.mnasnet import _get_depths
from .keypoint_detector import KeypointDetector


class MNASNetBackbone(_MNASNet):
    def forward(self, x):
        outs = []
        for f in self.layers:
            y = f(x)
            if x.shape[2] != y.shape[2]:
                outs.append(x)
            x = y
        outs.append(x)
        return outs


class MNASNet(KeypointDetector):
    stride = 4

    depth = {
        0.5: [16, 24, 48, 160],
        1.0: [24, 40, 96, 320],
    }

    def __init__(self, backbone, num_classes, alpha):
        super().__init__()

        backbone.__class__ = MNASNetBackbone
        del backbone.layers[-3:]
        del backbone.classifier
        self.backbone = backbone

        d = self.depth[alpha]

        self.layer4 = ConvBNReLU(d[-1], d[-2], 3, 1)
        self.layer3 = ConvBNReLU(d[-2] * 2, d[-3], 3, 1)
        self.layer2 = ConvBNReLU(d[-3] * 2, d[-4], 3, 1)
        self.layer1 = ConvBNReLU(d[-4] * 2, d[-4], 3, 1)
        self.fc = nn.Conv2d(d[-4], num_classes, 1, 1, 0)

    def _forward(self, x):
        outs = self.backbone(x)
        l1, l2, l3, l4 = outs[2:]  # stride = 4

        x = self.layer4(l4)
        x = F.interpolate(x, l3.shape[2:])

        x = torch.cat((x, l3), dim=1)
        x = self.layer3(x)
        x = F.interpolate(x, l2.shape[2:])

        x = torch.cat((x, l2), dim=1)
        x = self.layer2(x)
        x = F.interpolate(x, l1.shape[2:])

        x = torch.cat((x, l1), dim=1)
        x = self.layer1(x)

        x = self.fc(x)

        return x
