from torch import nn


class KeypointDetector(nn.Module):
    def __init__(self):
        super().__init__()
        assert hasattr(self, "stride"), "stride is necessary "

    def forward(self, x):
        y = self._forward(x)

        assert (
            x.shape[2] // self.stride == y.shape[2]
        ), f"stride={self.stride} doesn't much {x.shape[2]} => {y.shape[2]}"
        assert (
            x.shape[3] // self.stride == y.shape[3]
        ), f"stride{self.stride} doesn't much {x.shape[3]} => {y.shape[3]}"

        return y
