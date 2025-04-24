import torch
from torch import nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, set1, set2):
        if len(set1.shape) == 2:
            set1 = set1.unsqueeze(0)
        if len(set2.shape) == 2:
            set2 = set2.unsqueeze(0)
        assert set1.shape[0] == set2.shape[0] and set1.shape[2] == set2.shape[2], "Mismatch in batch or channels"

        set1 = set1.unsqueeze(2)  # B x M x 1 x C
        set2 = set2.unsqueeze(1)  # B x 1 x N x C
        distances = torch.norm(set1 - set2, dim=-1, p=2)  # B x M x N

        min_set1 = torch.min(distances, dim=2)[0]  # B x M
        min_set2 = torch.min(distances, dim=1)[0]  # B x N

        loss = torch.mean(min_set1, dim=1) + torch.mean(min_set2, dim=1)  # B
        if loss.shape[0] == 1:
            loss = loss.squeeze(0)
        return loss
