import torch
import torch.nn as nn


class DynamicBCEMSE(nn.Module):
    def __init__(self, use_sigmoid=False, mse_weight=16, bce_weight=1, alpha=1/6, reduction='mean'):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.alpha = alpha
        assert use_sigmoid==False, 'now use sigmoid is not supported'
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        loss = self.BCE_loss(pred, target) * (self.bce_weight * self.alpha) + self.mse_loss(pred_sigmoid, target) * (self.mse_weight * (1 - self.alpha))
        return loss
