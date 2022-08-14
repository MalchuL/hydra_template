import torch
import torch.nn as nn


class TClip(nn.Module):
    def __init__(self, min=0, max=1, weight=1):
        super().__init__()
        self.weight = weight
        self.min = min
        self.max = max
        self.loss = nn.L1Loss()

    def forward(self, tensor):
        max_mask = torch.clip(tensor, max=self.max)
        min_mask = torch.clip(tensor, min=self.min)
        loss = self.loss(tensor, max_mask) + self.loss(tensor, min_mask)
        return loss * self.weight

