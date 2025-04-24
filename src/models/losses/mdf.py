import torch
from torch import nn

class MDFLoss(nn.Module):
    def __init__(self):
        super(MDFLoss, self).__init__()

    def forward(self, set1, set2):
        if len(set1.shape) == 2:
            set1 = set1.unsqueeze(0)
        if len(set2.shape) == 2:
            set2 = set2.unsqueeze(0)
        assert set1.shape == set2.shape, "MDF requires same shape (B, N, C)"

        direct = torch.norm(set1 - set2, dim=2).mean(dim=1)  # B
        flipped = torch.norm(set1 - torch.flip(set2, dims=[1]), dim=2).mean(dim=1)  # B

        loss = torch.min(direct, flipped)
        if loss.shape[0] == 1:
            loss = loss.squeeze(0)
        return loss
