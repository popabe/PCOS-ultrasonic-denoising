import torch
import torch.nn as nn

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean'):
        super(CustomSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
