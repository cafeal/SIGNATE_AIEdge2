import torch
from torch import nn


class FocalLoss(nn.Module):
    def forward(self, pred, target):
        pos_inds = target.ge(1.0).float()
        neg_inds = target.lt(1.0).float()

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = - neg_loss
        else:
            loss = - (pos_loss + neg_loss) / num_pos
        return loss
