import torch


class Extractor(torch.nn.Module):
    @torch.jit.ignore
    def check_output(self, outs):
        for out, c in zip(outs, self.out_channels):
            assert out.shape[1] == c, "channel of out:{out.shape} != c: {c}"

    def forward(self, x):
        outs = self._forward_impl(x)

        if self.training:
            self.check_output(outs)
        return outs

    @classmethod
    def setup(cls, model):
        model.__class__ = cls
        model.remove_unused()
        return model
