import os
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils.configs import GeneratorType, DataDict, Hausdorff, LossNames
from src.models.diff_hausdorf import HausdorffLoss
from src.models.losses.chamfer import ChamferLoss
from src.models.losses.point import PointwiseMSELoss
from src.models.losses.mdf import MDFLoss

class Loss3D(nn.Module):

    def __init__(self, cfg, wm_mask_path=None):
        super(Loss3D, self).__init__()


        self.generator_type = cfg.generator_type
        self.use_traversability = cfg.use_traversability 
        self.collision_distance = 0.09

        self.target_dis = nn.MSELoss(reduction="mean")
        self.distance = HausdorffLoss(mode=cfg.distance_type)
        self.chamfer_loss = ChamferLoss()
        self.pointwise_mse_loss = PointwiseMSELoss()
        self.mdf_loss = MDFLoss()

        self.distance_type = cfg.distance_type
        self.scale_waypoints = 10.0
        self.last_ratio = cfg.last_ratio
        self.distance_ratio = cfg.distance_ratio
        self.traversability_ratio = cfg.traversability_ratio

        self.map_resolution = 1
        self.map_range = cfg.map_range 
        self.output_dir = cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    #                Core 3D Collisions / Traversability
    # ----------------------------------------------------------------
    def _cropped_distance_3d(self, path_vox, obstacle_vox):
        """
        path_vox:     [N, 3] integer voxel coords for predicted points
        obstacle_vox: [M, 3] integer voxel coords for obstacles (where mask==False)
        
        Returns:
          (loss_val, mean_dist)
        """
        # shape: obstacle_vox -> [M,1,3], path_vox -> [1,N,3]
        obs_reshape = obstacle_vox.view(-1, 1, 3).float()
        path_reshape = path_vox.view(1, -1, 3).float()

        # Broadcasting difference -> [M, N, 3]
        diff = obs_reshape - path_reshape
        dist = torch.norm(diff, p=2, dim=-1)  # [M, N]

        # distance to nearest obstacle for each path point => [N]
        min_dist_each_point = dist.min(dim=0)[0] * self.map_resolution

        # clamp distances, identify collisions
        traversability = torch.clamp(min_dist_each_point, 0.0001, self.collision_distance)
        violating_points = traversability[traversability < self.collision_distance]

        if len(violating_points) < 1:
            # No collisions => no penalty
            loss_val = torch.tensor(0.0, device=traversability.device, dtype=torch.float)
            mean_dist = torch.tensor(1.0, device=traversability.device, dtype=torch.float)
        else:
            # big penalty for points that are inside the collision_distance
            loss_val = torch.arctanh(
                (self.collision_distance - violating_points) / self.collision_distance
            ).mean()
            mean_dist = violating_points.mean()

        return loss_val, mean_dist

    def _local_collision_3d(self, yhat_3d, wm_mask):
        """
        yhat_3d: [B, N, 3] in world coords (x,y,z).
        wm_mask: [D, W, H] torch.bool => True is WM, False is outside WM.

        We'll treat anything outside WM => "obstacle".
        Return => (all_losses, all_traversabilities) each [B].
        """
        B, N, C = yhat_3d.shape
        assert C == 3, f"Expected 3D coords, got shape {yhat_3d.shape}."

        # indices => [M, 3], each row is an obstacle voxel
        # i.e., everywhere mask==False
        obstacle_inds = torch.stack(torch.where(wm_mask == False), dim=1)

        all_losses, all_traversabilities = [], []
        for i in range(B):
            coords_vox = self._world_to_voxel(yhat_3d[i], wm_mask)
            loss_val, t_val = self._cropped_distance_3d(coords_vox, obstacle_inds)
            all_losses.append(loss_val)
            all_traversabilities.append(t_val)

        all_losses_tensor = torch.stack(all_losses)
        all_traversabilities_tensor = torch.stack(all_traversabilities)
        return all_losses_tensor, all_traversabilities_tensor

    def _world_to_voxel(self, coords_xyz, wm_mask):
        """
        coords_xyz: [N, 3] in world coords (x, y, z).
        This function does a simplistic transform if your voxel spacing is 1mm, no offset.
        In reality, you might need to handle affines or bounding checks.

        Returns => [N, 3] integer voxel indices (z, y, x).
        """
        # If each voxel is 1 mm, we can assume coords ~ voxel
        # But you probably have an affine or spacing; 
        # you could store that in input_dict or read from wm_mask's nibabel header.
        # For now, let's assume it's 1mm & no offset.

        # We'll guess that your data is stored as (z, y, x).
        # If your coords are (x, y, z) we do the flip:
        coords_vox_z = coords_xyz[:, 2]  # z
        coords_vox_y = coords_xyz[:, 1]  # y
        coords_vox_x = coords_xyz[:, 0]  # x

        coords_vox = torch.stack([coords_vox_z, coords_vox_y, coords_vox_x], dim=1)
        coords_vox = torch.round(coords_vox).long()

        # Optionally clamp to ensure we don't go OOB
        D, W, H = wm_mask.shape  # e.g. D=Z, W=Y, H=X
        coords_vox[:, 0] = coords_vox[:, 0].clamp(0, D-1)
        coords_vox[:, 1] = coords_vox[:, 1].clamp(0, W-1)
        coords_vox[:, 2] = coords_vox[:, 2].clamp(0, H-1)

        return coords_vox

    def forward(self, input_dict):

        ygt = input_dict[DataDict.points]     # shape [B, N, 3]
        y_hat = input_dict[DataDict.prediction]
        subject_id = input_dict[DataDict.subject_id][0]

        # print("Shape of groundtruth: ", ygt.shape)
        # print("Shape of prediction: ", y_hat.shape)
        output = {}
        y_hat_poses = y_hat

        if self.use_traversability:
            B = y_hat_poses.shape[0]
            half_B = int(B / 2)
            # e.g. if you want half for collision checks
            traversability_hat_poses = y_hat_poses[half_B:]
            y_hat_poses = y_hat_poses[:half_B]
            ygt = ygt[:half_B]  # to match shape

        path_dis = self.distance(ygt, y_hat_poses).mean()
        final_path_dis = path_dis

        # Calculate sum of distances between consecutive points in y_hat_poses
        point_diffs = y_hat_poses[:, 1:, :] - y_hat_poses[:, :-1, :]
        segment_lengths = torch.norm(point_diffs, p=2, dim=2)
        total_segment_length_per_path = torch.sum(segment_lengths, dim=1)
        mean_total_segment_length = total_segment_length_per_path.mean()
        
        last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
        # first_pose_dis = self.target_dis(ygt[:, 0, :], y_hat_poses[:, 0, :])

        all_points_mse = self.target_dis(ygt, y_hat_poses)
        all_points_mse_mean = all_points_mse.mean()

        # all_loss = self.distance_ratio * final_path_dis + 2.0 * last_pose_dis + 2.0 * all_points_mse_mean + 10.0 * mean_total_segment_length

        all_loss = self.distance_ratio * path_dis +  2.0 * last_pose_dis + 20.0 * mean_total_segment_length
        output.update({
            LossNames.path_dis: final_path_dis,
            LossNames.last_dis: last_pose_dis,
        })

        if self.use_traversability:
            sub_id = subject_id
            wm_mask_path = f"/med/TractoDiff/data/trainset/{sub_id}/{sub_id}-generated_approximated_mask.nii.gz"
            
            if not os.path.exists(wm_mask_path):
                raise FileNotFoundError(f"WM mask not found at {wm_mask_path}")
            
            wm_nifti = nib.load(wm_mask_path)
            wm_data = wm_nifti.get_fdata()
            wm_mask_torch = torch.from_numpy(wm_data).bool()
            wm_mask_torch = wm_mask_torch.to(y_hat_poses.device)

            collision_loss, _ = self._local_collision_3d(traversability_hat_poses, wm_mask_torch)
            collision_loss_mean = collision_loss.mean().float()

            all_loss += self.traversability_ratio * collision_loss_mean
            output.update({LossNames.traversability: collision_loss_mean})

        output.update({LossNames.loss: all_loss})
        return output
    
    @torch.no_grad()
    def evaluate(self, input_dict, indices=0):
        ygt = input_dict[DataDict.points]
        y_hat = input_dict[DataDict.prediction]

        # Visualize 3D streamlines
        if DataDict.bundle in input_dict:
            subject_id = input_dict[DataDict.subject_id][0]
            bundle = input_dict[DataDict.bundle][0]

            for idx in range(len(y_hat)):
                vis_file = os.path.join(self.output_dir, f"streamline_vis_{subject_id}_{bundle}_{indices}_{idx}.png")
                visualize_3d_streamlines(
                    predictions=y_hat[idx].detach().cpu().numpy(),
                    ground_truth=ygt[idx].detach().cpu().numpy(),
                    subject_id=subject_id,
                    bundle=bundle,
                    split="testset",
                    output_file=vis_file
                )

        path_dis = self.distance(ygt, y_hat).mean()
        final_path_dis = path_dis

        last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat[:, -1, :])
        first_pose_dis = self.target_dis(ygt[:, 0, :], y_hat[:, 0, :])
        output = {
            LossNames.evaluate_last_dis: last_pose_dis,
            LossNames.evaluate_path_dis: final_path_dis,
        }

        if self.use_traversability:
            subject_id = input_dict[DataDict.subject_id][0]
            wm_mask_path = f"/med/TractoDiff/data/testset/{subject_id}/{subject_id}-generated_approximated_mask.nii.gz"

            wm_nifti = nib.load(wm_mask_path)
            wm_data = wm_nifti.get_fdata()
            wm_mask_torch = torch.from_numpy(wm_data).bool().to(y_hat.device)

            traversability_loss, traversability_values = self._local_collision_3d(y_hat,wm_mask_torch)
            traversability_loss_mean = traversability_loss.mean()
            output.update({LossNames.evaluate_traversability: traversability_loss_mean})
        
        return output

    def consistency_loss(self, output_dict, teacher_model=True, num_scales=40):
        """
        Compute consistency distillation loss
        Args:
            output_dict: Model output dictionary
            teacher_model: Whether to use teacher model predictions (True) or target model (False)
            num_scales: Number of noise scales to use
        Returns:
            Dictionary of loss values
        """
        # Get predictions
        student_pred = output_dict[DataDict.prediction]
        
        if teacher_model:
            reference_pred = output_dict["teacher_prediction"]
        else:
            reference_pred = output_dict["target_prediction"]
        
        # Compute L2 loss between student and reference predictions
        consistency_loss = torch.mean((student_pred - reference_pred) ** 2)
        
        # Combine with any other relevant losses from your existing framework
        loss_dict = {
            LossNames.consistency_loss: consistency_loss,
            LossNames.loss: consistency_loss  # Main loss for backward
        }
        
        return loss_dict


def visualize_3d_streamlines(predictions, ground_truth, subject_id, bundle, split="testset", output_file=None, context_tractogram=None):
    """
    Create a 3D visualization of predicted and ground truth streamlines.
    
    Args:
        predictions: [N, 3] array of predicted streamline points
        ground_truth: [N, 3] array of ground truth streamline points
        subject_id: Subject ID for loading original tractogram
        bundle: Bundle name
        split: Data split ('trainset', 'testset')
        output_file: Path to save the output image
        context_tractogram: Optional pre-loaded tractogram for context
    
    Returns:
        fig: matplotlib Figure object
    """
    # Create the figure for 3D plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Load original tractogram if not provided
    if context_tractogram is None:
        tract_path = f"/med/TractoDiff/data/{split}/{subject_id}/tractography/{subject_id}__{bundle}.trk"

        # "/med/TractoDiff/data/trainset/sub-1030/tractography/sub-1030__AF_L.trk"
        if os.path.exists(tract_path):
            try:
                tractogram = nib.streamlines.load(tract_path)
                context_streamlines = tractogram.streamlines
            except Exception as e:
                print(f"Error loading tractogram: {e}")
                context_streamlines = []
        else:
            print(f"Tractogram file not found: {tract_path}")
            context_streamlines = []
    else:
        context_streamlines = context_tractogram
    
    # Plot context streamlines with transparency (only plot a subset to avoid overcrowding)
    num_context = min(50, len(context_streamlines)) if hasattr(context_streamlines, '__len__') else 0
    for i in range(num_context):
        streamline = context_streamlines[i]
        # Plot with low opacity gray
        ax.plot3D(
            streamline[:, 0], 
            streamline[:, 1], 
            streamline[:, 2], 
            color='gray', 
            alpha=0.1, 
            linewidth=0.5
        )
    
    # Plot ground truth points (green)
    ax.scatter(
        ground_truth[:, 0], 
        ground_truth[:, 1], 
        ground_truth[:, 2], 
        color='green', 
        s=30, 
        label='Ground Truth'
    )
    
    # Plot predicted points (red)
    ax.scatter(
        predictions[:, 0], 
        predictions[:, 1], 
        predictions[:, 2], 
        color='red', 
        s=30, 
        label='Predicted'
    )
    
    # Find max dimensions to set equal aspect ratio
    all_points = np.vstack([predictions, ground_truth])
    x_range = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))
    y_range = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
    z_range = (np.min(all_points[:, 2]), np.max(all_points[:, 2]))
    
    # Calculate center and max range
    x_center = (x_range[0] + x_range[1]) / 2
    y_center = (y_range[0] + y_range[1]) / 2
    z_center = (z_range[0] + z_range[1]) / 2
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    
    # Set limits to be equal in all dimensions
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Streamline Visualization - {subject_id} {bundle}")
    
    # Add legend
    ax.legend()
    
    # Set a view that clearly shows the 3D structure
    ax.view_init(elev=20, azim=30)
    
    # Save or display
    if output_file:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return output_file
    else:
        return fig
