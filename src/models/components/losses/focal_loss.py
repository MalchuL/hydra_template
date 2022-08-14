import math

import numpy as np
import torch
import torch.nn as nn


def logit(p):
    return math.log(p) - math.log(1 - p)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.eps = 1e-4
        self.inv_eps = 1 - 1e-4
        self.logit_eps = logit(self.eps)
        self.logit_inv_eps = logit(self.inv_eps)

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred = torch.clip(pred, min=self.logit_eps, max=self.logit_inv_eps)
        loss = self.alpha * self.BCE_loss(pred, target) * self.mse_loss(pred_sigmoid, target)
        return loss

# if __name__ == '__main__':
#     x = 0.0001
#     y = logit(x)
#     print(y, sigmoid(y))