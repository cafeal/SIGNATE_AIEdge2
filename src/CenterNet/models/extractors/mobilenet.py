from torchvision.models import MobileNetV2 as _MobileNetV2
from .extractor import Extractor


class MobileNetV2Extractor(Extractor, _MobileNetV2):
    out_channels = [1280, 96, 32, 24]

    def _forward_impl(self, x):
        outs = []
        n_downsample = 0
        for f in self.features:
            y = f(x)
            if x.shape[2] != y.shape[2]:
                if n_downsample >= 2:
                    outs.append(x)
                n_downsample += 1
            x = y
        outs.append(x)

        return outs[::-1]

    def remove_unused(self):
        del self.classifier
