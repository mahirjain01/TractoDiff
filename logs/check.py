import torch
from torch import nn
from typing import Any # Added for type hinting consistency

# Define a simple placeholder for the Hausdorff enum/config if not available
# In a real scenario, this would come from src.utils.configs
class Hausdorff:
    average = 'average'
    max = 'max'

class HausdorffLoss(nn.Module):
    """
    Calculates the Hausdorff Distance between two point sets.
    """
    def __init__(self, mode=Hausdorff.average):
        """
        Initializes the HausdorffLoss module.

        Args:
            mode (str): The mode for Hausdorff calculation ('average' or 'max').
                        Defaults to 'average'.
        """
        super(HausdorffLoss, self).__init__()
        self.mode = mode
        # Validate the mode
        if mode not in [Hausdorff.average, Hausdorff.max]:
             # Raise error if mode is invalid, providing valid options
             raise ValueError(f"Invalid mode '{mode}'. Choose from '{Hausdorff.average}' or '{Hausdorff.max}'.")


    def forward(self, set1, set2):
        """
        Calculates the forward pass of the Hausdorff distance.

        Args:
            set1 (torch.Tensor): The first set of points (e.g., ground truth).
                                 Shape: (B, M, C) or (M, C).
            set2 (torch.Tensor): The second set of points (e.g., prediction).
                                 Shape: (B, N, C) or (N, C).

        Returns:
            torch.Tensor: The calculated Hausdorff distance(s) for the batch.
                          Shape: (B,) or scalar if B=1.
        """
        # Ensure inputs are 3D tensors (Batch x Points x Coordinates)
        if len(set1.shape) == 2:
            set1 = set1.unsqueeze(0) # Add batch dimension if missing
        if len(set2.shape) == 2:
            set2 = set2.unsqueeze(0) # Add batch dimension if missing

        # Assertions to check shape compatibility
        assert len(set1.shape) == 3 and len(set2.shape) == 3, \
            f"Input tensors must be 3D (B x N x C). Got shapes: {set1.shape} and {set2.shape}"
        assert set1.shape[0] == set2.shape[0] and set1.shape[2] == set2.shape[2], \
            f"Batch size (B) and coordinate dimensions (C) must match. Got shapes: {set1.shape} and {set2.shape}"

        # Prepare tensors for pairwise distance calculation via broadcasting
        # set1: (B, M, C) -> (B, M, 1, C)
        # set2: (B, N, C) -> (B, 1, N, C)
        set1_expanded = set1.unsqueeze(2)
        set2_expanded = set2.unsqueeze(1)

        # Calculate pairwise Euclidean distances
        # (B, M, 1, C) - (B, 1, N, C) -> (B, M, N, C) -> norm -> (B, M, N)
        distances = torch.norm(set1_expanded - set2_expanded, dim=-1, p=2)

        # Calculate Hausdorff distance based on the specified mode
        if self.mode == Hausdorff.average:
            # Average Hausdorff: mean of min distances in both directions
            # Find min distance from each point in set1 to set2
            min_dist_1_to_2, _ = torch.min(distances, dim=2) # Shape: (B, M)
            term_1 = torch.mean(min_dist_1_to_2, dim=1)      # Shape: (B,)

            # Find min distance from each point in set2 to set1
            min_dist_2_to_1, _ = torch.min(distances, dim=1) # Shape: (B, N)
            term_2 = torch.mean(min_dist_2_to_1, dim=1)      # Shape: (B,)

        elif self.mode == Hausdorff.max:
            # Max (Directed) Hausdorff: max of min distances in both directions
             # Find min distance from each point in set1 to set2
            min_dist_1_to_2, _ = torch.min(distances, dim=2) # Shape: (B, M)
            term_1, _ = torch.max(min_dist_1_to_2, dim=1)      # Shape: (B,)

             # Find min distance from each point in set2 to set1
            min_dist_2_to_1, _ = torch.min(distances, dim=1) # Shape: (B, N)
            term_2, _ = torch.max(min_dist_2_to_1, dim=1)      # Shape: (B,)
        else:
             # This case should not be reached due to __init__ validation,
             # but included for robustness.
            raise ValueError(f"Hausdorff mode '{self.mode}' is not defined.")

        # Final Hausdorff distance is the sum of the two terms
        res = term_1 + term_2

        # Remove batch dimension if it was originally 1
        if res.shape[0] == 1:
            res = res.squeeze(0)

        return res

# --- Input Data ---
# Prediction Tensor
pred = torch.tensor([[0.0724, 0.0561, 0.2171],
                     [0.0761, 0.0487, 0.2092],
                     [0.0793, 0.0424, 0.1999],
                     [0.0815, 0.0381, 0.1931],
                     [0.0830, 0.0348, 0.1880],
                     [0.0839, 0.0322, 0.1842],
                     [0.0845, 0.0300, 0.1812],
                     [0.0848, 0.0281, 0.1787],
                     [0.0849, 0.0265, 0.1767],
                     [0.0849, 0.0251, 0.1750],
                     [0.0848, 0.0239, 0.1736],
                     [0.0847, 0.0228, 0.1724],
                     [0.0845, 0.0219, 0.1713],
                     [0.0842, 0.0211, 0.1703],
                     [0.0840, 0.0203, 0.1694],
                     [0.0837, 0.0196, 0.1686]], dtype=torch.float32) # Ensure float dtype

# Ground Truth Tensor
gt = torch.tensor([[-46.3664, -20.6757,  17.3023],
                   [-45.7891, -21.0087,  17.7005],
                   [-41.5682, -19.7189,  19.2175],
                   [-38.2632, -19.5048,  20.1944],
                   [-34.4050, -20.5902,  25.4589],
                   [-33.9115, -20.7160,  26.6938],
                   [-30.9884, -20.4352,  29.8560],
                   [-29.9695, -20.0444,  31.4647],
                   [-28.5531, -18.3597,  34.4458],
                   [-26.5023, -14.4187,  38.5789],
                   [-25.2054, -12.0238,  40.1230],
                   [-24.8447, -10.3351,  40.9723],
                   [-25.2342,  -8.3844,  41.7666],
                   [-25.1877,  -3.4272,  42.8450],
                   [-24.7864,  -0.1839,  43.3349],
                   [-24.5932,   3.1318,  43.1050]], dtype=torch.float32) # Ensure float dtype


distance_calculator = HausdorffLoss(mode=Hausdorff.average)

# Calculate the distance
# Note: The tensors already have requires_grad=False by default.
# If your original tensors had gradients, detach them first:
# path_dis = distance_calculator(gt.detach(), pred.detach())
path_dis = distance_calculator(gt, pred)

# --- Output ---
print(f"Input shapes: gt={gt.shape}, pred={pred.shape}")
print(f"Hausdorff Distance (Mode: {distance_calculator.mode}):")
print(path_dis)

# Optional: Print the tensor value if you don't need the tensor object itself
# print(path_dis.item())
