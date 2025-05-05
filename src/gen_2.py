#!/usr/bin/env python

import argparse
import numpy as np
import torch
import nibabel as nib

from nibabel.streamlines.tractogram import LazyTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.tracking.metrics import length, mean_curvature
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import (get_reference_info,
                           create_tractogram_header)
# from dipy.core.geometry import angle_between_vectors
import os
import sys
from warnings import warn
import time  # Add time module

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, '/tracto/TrackToLearn')
sys.path.insert(0, '/tracto')

from src.models.model import get_model
from src.utils.configs import DataDict, TrainingConfig
from src.loss_3d import Loss3D, visualize_3d_streamlines
from environments.env import BaseEnv


def generate_condition_vector(point: np.ndarray, env: BaseEnv) -> np.ndarray:
    """Project‑specific function provided by user – do **not** edit."""
    point = np.asarray(point).reshape(1, 1, 3)
    return env._format_state(point)[0]

class StreamlineGenerator:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.iteration = 0
        self.epoch = 0
        self.training = False
        self.output_dir = cfg.output_dir

        self.device = "cuda:1"        
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        print("The device is: ", self.device)
            
        # Initialize model
        self.model = get_model(config=cfg.model, device=self.device)
        
        # Setup loss function
        self.loss_func = Loss3D(cfg=cfg.loss)
        self.loss_func = self.loss_func.to(self.device)

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=self.device)

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            warn('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn('Unexpected keys: {}'.format(unexpected_keys))
        print('Model has been loaded.')
        return state_dict

    def load_learning_parameters(self, state_dict):
        # For inference, we might only want to keep track of which epoch/iteration 
        # the model was saved from
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            print('Model was saved at epoch: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print('Model was saved at iteration: {}.'.format(self.iteration))

    def set_eval_mode(self):
        """Set the model to evaluation mode"""
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def _ensure_model_on_device(self):
        """Helper method to ensure model is on the correct device"""
        if hasattr(self.model, 'device'):
            if str(self.model.device) != str(self.device):
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

    def load_wm_mask(self, wm_loc):
        """Load white matter mask from NIFTI file."""
        wm_img = nib.load(wm_loc)
        wm_data = wm_img.get_fdata()
        return wm_data, wm_img.affine

    def check_termination_conditions(self, streamline, wm_mask, wm_affine, max_length=200, max_angle=60, min_fa=0.2):
        """
        Check if streamline should be terminated based on various criteria.
        """
        # Length check
        if len(streamline) >= max_length:
            return True
        
        # Boundary check (convert last point to voxel coordinates)
        last_point = streamline[-1]
        vox_coords = np.linalg.inv(wm_affine).dot(np.append(last_point, 1))[:3]
        vox_coords = np.round(vox_coords).astype(int)
        
        # Check if point is outside WM mask
        if (vox_coords[0] < 0 or vox_coords[0] >= wm_mask.shape[0] or
            vox_coords[1] < 0 or vox_coords[1] >= wm_mask.shape[1] or
            vox_coords[2] < 0 or vox_coords[2] >= wm_mask.shape[2] or
            wm_mask[vox_coords[0], vox_coords[1], vox_coords[2]] == 0):
            return True
        
        return False

    def generate_streamline(self, seed_point, subject_id, bundle, dataset_file, wm_loc):
        """
        Generate a complete streamline starting from a seed point.
        """
        # Load WM mask
        wm_mask, wm_affine = self.load_wm_mask(wm_loc)
        
        # Initialize streamline with seed point
        streamline = [seed_point]
        current_point = seed_point
        
        while True:
            # Generate condition vector for current point
            condition_vector = generate_condition_vector(point=current_point,env=self.env)
            
            # Convert to tensor and add batch dimension
            condition_vector = torch.from_numpy(condition_vector).float().unsqueeze(0).to(self.device)

            input_dict = {DataDict.condition: condition_vector}
            
            # Generate next segment using model
            with torch.no_grad():
                output = self.model(input_dict=input_dict, sample=True)
                new_segment = output[DataDict.prediction].cpu().numpy()[0]  # Get first batch element
            
            # Add new segment to streamline
            streamline.extend(new_segment)
            
            # Update current point to last point of new segment
            current_point = new_segment[-1]
            
            # Check termination conditions
            if self.check_termination_conditions(np.array(streamline), wm_mask, wm_affine):
                break
        
        return np.array(streamline)

    def generate_all_streamlines(self, args):
        """Generate all streamlines and save visualizations"""
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.output_trk), exist_ok=True)
        
        # Ensure output_trk has .trk extension
        if not args.output_trk.endswith('.trk'):
            args.output_trk = os.path.join(os.path.dirname(args.output_trk), 
                                         f"{os.path.splitext(os.path.basename(args.output_trk))[0]}.trk")
        
        # Create visualization subdirectory
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load model from snapshot
        print(f"Loading model from {args.model_path}")
        state_dict = self.load_snapshot(args.model_path)
        self.load_learning_parameters(state_dict)
        self.set_eval_mode()
        self._ensure_model_on_device()

        self.seed_img = nib.load(args.wm_loc)

        self.env = BaseEnv(
            dataset_file=args.dataset_file,
            wm_loc=args.wm_loc,
            subject_id=args.subject,
            n_signal=1,
            n_dirs=8,
            step_size=0.2,
            max_angle=60,
            min_length=10,
            max_length=200,
            n_seeds_per_voxel=4,
            rng=np.random.RandomState(1337),
            add_neighborhood=1.5,
            compute_reward=True,
            device=self.device,
        )
        
        # Load seed streamlines
        print(f"Loading seed streamlines from {args.seed_trk}")
        seed_tractogram = load_tractogram(args.seed_trk, reference='same', bbox_valid_check=False, to_space=Space.RASMM)
        seed_streamlines = seed_tractogram.streamlines
        print(f"Loaded {len(seed_streamlines)} seed streamlines")
        
        # Initialize timing variables
        total_start_time = time.time()
        streamline_times = []
          
        # Generate new streamlines
        generated_streamlines = []
        for i, seed_streamline in enumerate(seed_streamlines[:100]): 
            print(f"Generating streamline {i+1}/{100}")
            streamline_start_time = time.time()
            
            # Use first point as seed
            seed_point = seed_streamline[0]
            
            # Generate new streamline
            new_streamline = self.generate_streamline(
                seed_point=seed_point,
                subject_id=args.subject,
                bundle=args.bundle,
                dataset_file=args.dataset_file,
                wm_loc=args.wm_loc
            )
            
            generated_streamlines.append(new_streamline)
            
            # Calculate time for this streamline
            streamline_time = time.time() - streamline_start_time
            streamline_times.append(streamline_time)
            print(f"Time taken for streamline {i+1}: {streamline_time:.2f} seconds")
        
        # Calculate total time
        total_time = time.time() - total_start_time

        data_per_streamlines = {}
        # tractogram = LazyTractogram(lambda: generated_streamlines,
        #                             data_per_streamlines,
        #                             affine_to_rasmm=self.seed_img.affine)

        sft = StatefulTractogram(
            Streamlines(generated_streamlines),       # streamlines
            self.seed_img,                     # same affine as the seed NIfTI
            Space.RASMM)
        
        # Create new tractogram and save
        print(f"Saving generated streamlines to {args.output_trk}")

        # reference = get_reference_info(self.seed_img)
        # header = create_tractogram_header(nib.streamlines.TrkFile, *reference)

        # new_tractogram = seed_tractogram
        # new_tractogram.streamlines = Streamlines(generated_streamlines)
        # nib.streamlines.save(tractogram, args.output_trk, header=header)

        save_tractogram(sft, args.output_trk, bbox_valid_check=False) 
       # Create final visualizations
        print("Generating final visualizations...")
        
        # Stack all generated streamlines for visualization
        all_generated = np.vstack(generated_streamlines)
        all_ground_truth = np.vstack(seed_streamlines)  # Match the number of generated streamlines
        
        # 1. Plot of only predicted streamlines
        pred_vis_file = os.path.join(vis_dir, "predicted_streamlines_vis.png")
        print(f"Saving predicted streamlines visualization to {pred_vis_file}")
        visualize_3d_streamlines(
            predictions=all_generated,
            ground_truth=all_generated,  # Use same data for both to show only predictions
            subject_id=args.subject,
            bundle=args.bundle,
            split="testset",
            output_file=pred_vis_file
        )
        
        # 2. Comparison plot of GT and generated streamlines
        comp_vis_file = os.path.join(vis_dir, "comparison_streamlines_vis.png")
        print(f"Saving comparison visualization to {comp_vis_file}")
        visualize_3d_streamlines(
            predictions=all_generated,
            ground_truth=all_ground_truth,
            subject_id=args.subject,
            bundle=args.bundle,
            split="testset",
            output_file=comp_vis_file
        )
        
        # Print timing summary
        print("=== Timing Summary ===")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per streamline: {np.mean(streamline_times):.2f} seconds")
        print(f"Fastest streamline: {min(streamline_times):.2f} seconds")
        print(f"Slowest streamline: {max(streamline_times):.2f} seconds")
        print(f"Standard deviation: {np.std(streamline_times):.2f} seconds")
        print("=====================")
        
        print("Streamline generation complete!")
        print(f"Generated TRK file saved as: {args.output_trk}")
        print(f"Visualizations saved in: {vis_dir}")
        print("  - predicted_streamlines_vis.png (predicted only)")
        print("  - comparison_streamlines_vis.png (GT vs predicted)")

def main():
    parser = argparse.ArgumentParser(description="Generate streamlines using diffusion model")
    
    # Required arguments
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--bundle", type=str, required=True, help="Bundle name")
    parser.add_argument("--seed_trk", type=str, required=True, help="Path to seed streamline file (.trk)")
    parser.add_argument("--output_trk", type=str, required=True, help="Path to output streamline file (.trk)")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to HDF5 dataset file")
    parser.add_argument("--wm_loc", type=str, required=True, help="Path to white matter mask")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create training config
    cfg = TrainingConfig
    cfg.name = "TractoDiff"
    cfg.model.type = "diffusion"
    cfg.model.generator_type = "diffusion"
    
    # Initialize generator and run
    generator = StreamlineGenerator(cfg)
    generator.generate_all_streamlines(args)

if __name__ == "__main__":
    main() 

     

</rewritten_file> 