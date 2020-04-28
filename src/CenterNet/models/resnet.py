import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet as _ResNet
from torchvision.models.resnet import BasicBlock
from .keypoint_detector import KeypointDetector


class ResNetBackbone(_ResNet):
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
            outs.append(x)
        return outs


class ResNet(KeypointDetector):
    stride = 4

    def __init__(
        self,
        backbone,
        layers,
        num_classes,
        block=BasicBlock,
        norm_layer=nn.BatchNorm2d,
    ):
        assert isinstance(backbone, _ResNet)

        super().__init__()
        self._norm_layer = norm_layer
        self._block = block

        backbone.__class__ = ResNetBackbone
        del backbone.avgpool
        del backbone.fc
        self.backbone = backbone

        self.layer4 = self._make_layer(512, 256, layers[3])
        self.layer3 = self._make_layer(256 + 256, 128, layers[2])
        self.layer2 = self._make_layer(128 + 128, 64, layers[1])
        self.layer1 = self._make_layer(64 + 64, 64, layers[0])

        self.fc = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1, 1, 0, bias=True),
        )

        self.fc[-1].bias.data.fill_(-2.19)

    def _make_layer(self, inplanes, planes, blocks):
        downsample = None
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False), self._norm_layer(planes),
            )

        layers = []
        layers.append(
            self._block(
                inplanes, planes, downsample=downsample, norm_layer=self._norm_layer,
            )
        )

        for _ in range(1, blocks):
            layers.append(self._block(planes, planes, norm_layer=self._norm_layer))

        layer = nn.Sequential(*layers)
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return layer
            

    def _forward(self, x):
        interpolation = "nearest"
        # interpolation = "bilinear"

        l1, l2, l3, l4 = self.backbone(x)

        x = self.layer4(l4)
        x = F.interpolate(x, l3.shape[2:], mode=interpolation)

        x = torch.cat((x, l3), dim=1)
        x = self.layer3(x)
        x = F.interpolate(x, l2.shape[2:], mode=interpolation)

        x = torch.cat((x, l2), dim=1)
        x = self.layer2(x)
        x = F.interpolate(x, l1.shape[2:], mode=interpolation)

        x = torch.cat((x, l1), dim=1)
        x = self.layer1(x)
        x = self.fc(x)

        return x
