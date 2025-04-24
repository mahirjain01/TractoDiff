import torch
from torch import nn

class PointwiseMSELoss(nn.Module):
    def __init__(self):
        super(PointwiseMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, set1, set2):
        if len(set1.shape) == 2:
            set1 = set1.unsqueeze(0)
        if len(set2.shape) == 2:
            set2 = set2.unsqueeze(0)
        assert set1.shape == set2.shape, "Set sizes must match for Pointwise MSE"

        loss = self.mse(set1, set2).mean(dim=[1, 2])  # B
        if loss.shape[0] == 1:
            loss = loss.squeeze(0)
        return loss
