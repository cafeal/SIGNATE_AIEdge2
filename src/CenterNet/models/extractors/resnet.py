from torchvision.models.resnet import ResNet as _ResNet
from .extractor import Extractor


class ResNetExtractor(Extractor, _ResNet):
    out_channels = [512, 256, 128, 64]

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)

        x = self.layer2(x)
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)

        x = self.layer4(x)
        outs.append(x)

        return outs[::-1]

    def remove_unused(self):
        del self.avgpool
        del self.fc
