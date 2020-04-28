import torch
import torch.nn as nn
import torch.nn.functional as F


class Detector(torch.nn.Module):
    def __init__(self, extractor):
        assert hasattr(self, "stride"), "stride is necessary "

        super().__init__()
        self.extractor = extractor

    @torch.jit.ignore
    def check_output(self, x, y):
        assert (
            x.shape[2] // self.stride == y.shape[2]
        ), f"stride={self.stride} doesn't much {x.shape[2]} => {y.shape[2]}"
        assert (
            x.shape[3] // self.stride == y.shape[3]
        ), f"stride{self.stride} doesn't much {x.shape[3]} => {y.shape[3]}"

    def forward(self, x):
        outs = self.extractor(x)
        y = self._forward(outs)

        if self.training:
            self.check_output(x, y)
        return y


class GeneralDetector(Detector):
    def __init__(self, extractor, num_classes):
        super().__init__(extractor)
        in_channels = extractor.out_channels

        prev_planes = 0
        layers = []
        for i in range(len(in_channels)):
            in_planes = in_channels[i]
            planes = in_channels[i + 1] if i < len(in_channels) - 1 else in_planes
            layer = self._make_layer(in_planes + prev_planes, planes, i)
            layers.append(layer)

            prev_planes = planes

        self.layers = nn.ModuleList(layers)
        self.fc = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, num_classes, 1, 1, 0, bias=True),
        )

        self.fc[-1].bias.data.fill_(-2.19)

    def _make_layer(self, in_planes, outplanes, layer_idx):
        raise NotImplementedError

    def _forward(self, outs):
        interpolation = "nearest"
        # interpolation = "bilinear"

        x = None
        for out, layer in zip(outs, self.layers):
            if x is not None:
                x = F.interpolate(x, out.shape[2:], mode=interpolation)
                out = torch.cat((x, out), dim=1)
            x = layer(out)

        x = self.fc(x)

        return x
