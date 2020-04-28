from torchvision.models import MNASNet as _MNASNet
from .extractor import Extractor


class MNASNetExtractor(Extractor, _MNASNet):
    def _forward_impl(self, x):
        outs = []
        n_downsample = 0
        for f in self.layers:
            y = f(x)
            if x.shape[2] != y.shape[2]:
                if n_downsample >= 2:
                    outs.append(x)
                n_downsample += 1
            x = y
        outs.append(x)
        return outs[::-1]

    @property
    def out_channels(self):
        if self.alpha == 0.5:
            return [160, 48, 24, 16]
        elif self.alpha == 1.0:
            return [320, 96, 40, 24]
        else:
            raise ValueError("alpha should be 0.5, 1.0")

    def remove_unused(self):
        del self.layers[-3:]
        del self.classifier
