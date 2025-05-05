import copy
import math
import os
import pickle
import shutil

import nibabel as nib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from os.path import join, exists
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba
import cv2

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

        self.train_poses = cfg.train_poses
        self.distance_type = cfg.distance_type
        self.scale_waypoints = 10.0
        self.last_ratio = cfg.last_ratio
        self.distance_ratio = cfg.distance_ratio
        self.traversability_ratio = cfg.traversability_ratio

        self.map_resolution = 1
        self.map_range = cfg.map_range 
        self.output_dir = cfg.output_dir
        if self.output_dir and not exists(self.output_dir):
            os.makedirs(self.output_dir)

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

    # ----------------------------------------------------------------
    #               Forward Methods (Diffusion / CVAE)
    # ----------------------------------------------------------------
    def forward_diffusion(self, input_dict):

        ygt = input_dict[DataDict.points]     # shape [B, N, 3]
        y_hat = input_dict[DataDict.prediction]
        subject_id = input_dict[DataDict.subject_id][0]

        # print("Shape of groundtruth: ", ygt.shape)
        # print("Shape of prediction: ", y_hat.shape)
        output = {}

        y_hat_poses = y_hat

        # if self.train_poses:
        #     y_hat_poses = y_hat * self.scale_waypoints
        # else:
        #     y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        if self.use_traversability:
            B = y_hat_poses.shape[0]
            half_B = int(B / 2)
            # e.g. if you want half for collision checks
            traversability_hat_poses = y_hat_poses[half_B:]
            y_hat_poses = y_hat_poses[:half_B]
            ygt = ygt[:half_B]  # to match shape

        path_dis = self.distance(ygt, y_hat_poses).mean()
        mdf_dis = self.mdf_loss(ygt, y_hat_poses).mean()

        final_path_dis = 0.7*path_dis + mdf_dis
        last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
        all_loss = self.distance_ratio * final_path_dis + self.last_ratio * last_pose_dis
        output.update({
            LossNames.path_dis: final_path_dis,
            LossNames.last_dis: last_pose_dis,
        })

        if self.use_traversability:
            sub_id = subject_id
            wm_mask_path = f"/tracto/TractoDiff/data/trainset/{sub_id}/{sub_id}-generated_approximated_mask_1mm.nii.gz"
            
            if not os.path.exists(wm_mask_path):
                raise FileNotFoundError(f"WM mask not found at {wm_mask_path}")
            
            wm_nifti = nib.load(wm_mask_path)
            wm_data = wm_nifti.get_fdata()
            wm_mask_torch = torch.from_numpy(wm_data).bool()
            wm_mask_torch = wm_mask_torch.to(y_hat_poses.device)

            collision_loss, traversability_vals = self._local_collision_3d(traversability_hat_poses, wm_mask_torch)
            collision_loss_mean = collision_loss.mean().float()

            all_loss += self.traversability_ratio * collision_loss_mean
            output.update({LossNames.traversability: collision_loss_mean})

        output.update({LossNames.loss: all_loss})
        return output

    def forward(self, input_dict):
        if self.generator_type == GeneratorType.cvae:
            return self.forward_cvae(input_dict=input_dict)
        elif self.generator_type == GeneratorType.diffusion:
            return self.forward_diffusion(input_dict=input_dict)
        else:
            raise ValueError("Unknown generator_type {}".format(self.generator_type))

    def convert_path_pixel(self, trajectory):
        return np.clip(np.around(trajectory / self.map_resolution)[:, :2] + self.map_range, 0, np.inf)

    def show_path_local_map(self, trajectory, gt_path, local_map, idx=0, indices=0):
        return write_png(local_map=local_map, center=np.array([local_map.shape[0] / 2, local_map.shape[1] / 2]),
                         file=join(self.output_dir, "local_map_trajectory_{}.png".format(indices + idx)),
                         paths=[self.convert_path_pixel(trajectory=trajectory)],
                         others=self.convert_path_pixel(trajectory=gt_path))

    @torch.no_grad()
    def evaluate(self, input_dict, indices=0):
        ygt = input_dict[DataDict.points]
        y_hat = input_dict[DataDict.prediction]

        y_hat_poses = y_hat

        # if self.train_poses:
        #     y_hat_poses = y_hat * self.scale_waypoints
        # else:
        #     y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        # print("y_hat is: ", y_hat[0])
        # print("y_hat_poses is: ", y_hat_poses[0])
        # print("Shape of groundtruth: ", ygt.shape)
        # print("Shape of prediction: ", y_hat.shape)

        if self.output_dir is not None:
            all_trajectories = input_dict[DataDict.all_trajectories]

            # Visualize 3D streamlines
            if DataDict.bundle in input_dict:
                subject_id = input_dict[DataDict.subject_id][0]
                bundle = input_dict[DataDict.bundle][0]

                # print("The predicted trajectory is: ", y_hat_poses[0])
                # print("The ground truth trajectory is: ", ygt[0])
                # Generate 3D visualization of the streamlines
                for idx in range(len(y_hat_poses)):
                    vis_file = join(self.output_dir, f"streamline_vis_{subject_id}_{bundle}_{indices}_{idx}.png")
                    visualize_3d_streamlines(
                        predictions=y_hat_poses[idx].detach().cpu().numpy(),
                        ground_truth=ygt[idx].detach().cpu().numpy(),
                        subject_id=subject_id,
                        bundle=bundle,
                        split="testset",
                        output_file=vis_file
                    )

            path_dis = self.distance(ygt, y_hat_poses).mean()
            mdf_dis = self.mdf_loss(ygt, y_hat_poses).mean()
            final_path_dis = 0.7*path_dis + 0.3*mdf_dis

            last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
            output = {
                LossNames.evaluate_last_dis: final_path_dis,
                LossNames.evaluate_path_dis: path_dis,
            }

            if self.use_traversability:
                subject_id = input_dict[DataDict.subject_id][0]
                wm_mask_path = f"/tracto/TractoDiff/data/testset/{subject_id}/{subject_id}-generated_approximated_mask_1mm.nii.gz"

                wm_nifti = nib.load(wm_mask_path)
                wm_data = wm_nifti.get_fdata()
                wm_mask_torch = torch.from_numpy(wm_data).bool().to(y_hat_poses.device)

                traversability_loss, traversability_values = self._local_collision_3d(y_hat_poses,wm_mask_torch)
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
        tract_path = f"/tracto/TractoDiff/data/{split}/{subject_id}/tractography/{subject_id}__{bundle}.trk"

        # "/tracto/TractoDiff/data/trainset/sub-1030/tractography/sub-1030__AF_L.trk"
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


def write_png(local_map=None, rgb_local_map=None, center=None, targets=None, paths=None, paths_color=None, path=None,
              crop_edge=None, others=None, file=None):
    """
    Create a 2D visualization of paths on a local map
    
    This function is maintained for backward compatibility with existing code
    For 3D visualization, use visualize_3d_streamlines instead
    """
    dis = 2
    x_range = [local_map.shape[0], 0]
    y_range = [local_map.shape[1], 0]
    if rgb_local_map is not None:
        local_map_fig = rgb_local_map
    else:
        local_map_fig = np.repeat(local_map[:, :, np.newaxis], 3, axis=2) * 255
    if center is not None:
        assert center.shape[0] == 2 and len(center.shape) == 1, "path should be 2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(center + np.array([x, y]))
        all_points = np.stack(all_points).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if targets is not None and len(targets) > 0:
        xs, ys = targets[:, 0], targets[:, 1]
        xs = np.clip(xs, dis, local_map_fig.shape[0] - dis)
        ys = np.clip(ys, dis, local_map_fig.shape[1] - dis)
        clipped_targets = np.stack((xs, ys), axis=-1)

        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(clipped_targets + np.array([x, y]))
        if len(clipped_targets.shape) == 2:
            all_points = np.concatenate(all_points, axis=0).astype(int)
        else:
            all_points = np.stack(all_points, axis=0).astype(int)

        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if others is not None:
        assert others.shape[1] == 2 and len(others.shape) == 2, "path should be Nx2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(others + np.array([x, y]))
        all_points = np.concatenate(all_points, axis=0).astype(int)

        xs, ys = all_points[:, 0], all_points[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 255
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 0

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if path is not None:
        assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
        all_pts = path
        all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                  all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                  all_pts), axis=0)
        xs, ys = all_pts[:, 0], all_pts[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 0
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 255

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if paths is not None:
        for p_idx in range(len(paths)):
            path = paths[p_idx]
            if len(path) == 1 or np.any(path[0] == np.inf):
                continue
            path = np.asarray(path, dtype=int)
            assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
            all_pts = path
            all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                      all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                      all_pts), axis=0)
            xs, ys = all_pts[:, 0], all_pts[:, 1]
            xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
            ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
            if paths_color is not None:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 0
                local_map_fig[xs, ys, 2] = paths_color[p_idx]
            else:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 255
                local_map_fig[xs, ys, 2] = 255

            if x_range[0] > min(all_pts[:, 0]):
                x_range[0] = min(all_pts[:, 0])
            if x_range[1] < max(all_pts[:, 0]):
                x_range[1] = max(all_pts[:, 0])
            if y_range[0] > min(all_pts[:, 1]):
                y_range[0] = min(all_pts[:, 1])
            if y_range[1] < max(all_pts[:, 1]):
                y_range[1] = max(all_pts[:, 1])
    if crop_edge:
        local_map_fig = local_map_fig[
                        max(0, x_range[0] - crop_edge):min(x_range[1] + crop_edge, local_map_fig.shape[0]),
                        max(0, y_range[0] - crop_edge):min(y_range[1] + crop_edge, local_map_fig.shape[1])]
    if file is not None:
        cv2.imwrite(file, local_map_fig)
    return local_map_fig