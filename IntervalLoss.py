import torch
import torch.nn as nn
import torch.nn.funcitonal as F

class IntervalLoss(nn.Module):
    def __init__(self, balance_weight=[0.5, 0.5]):
        super(IntervalLoss, self).__init__()
        self.original_loss = nn.BCEWithLogitsLoss()
        self.balance_weight = balance_weight
        return

    def forward(self, predict, ground_truth):
        bce_loss = self.original_loss(predict, ground_truth)

