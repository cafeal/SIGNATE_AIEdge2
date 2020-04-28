from efficientnet_pytorch import EfficientNet as _EfficientNet
from .extractor import Extractor


class EfficientNetExtractor(Extractor, _EfficientNet):
    def _forward_impl(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        outs = []
        n_downsample = 0
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)

            if x.shape[2] != y.shape[2]:
                if n_downsample >= 1:
                    outs.append(x)
                n_downsample += 1
            x = y

        x = self._swish(self._bn1(self._conv_head(x)))

        outs.append(x)

        return outs[::-1]

    @property
    def out_channels(self):
        return {
            "efficientnet-b0": [1280, 112, 40, 24],
            "efficientnet-b1": [1280, 112, 40, 24],
            "efficientnet-b2": [1408, 120, 48, 24],
            "efficientnet-b3": [1536, 136, 48, 32],
            "efficientnet-b4": [1792, 160, 56, 32],
            "efficientnet-b5": [2048, 176, 64, 40],
            "efficientnet-b6": [2304, 200, 72, 40],
            "efficientnet-b7": [2560, 224, 80, 48],
        }[self.model_name]

    def remove_unused(self):
        del self._avg_pooling
        del self._dropout
        del self._fc

    @classmethod
    def get_pretrained(cls, model_name):
        model = cls.from_pretrained(model_name)
        model = cls.setup(model)
        model.model_name = model_name
        return model
