#!/usr/bin/env python

import argparse
import numpy as np
import torch
import nibabel as nib
import os
import sys
import time
from warnings import warn
from tqdm import tqdm # Add tqdm for progress bars

# --- Add project paths (keep as is or adjust if needed) ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add other necessary paths if they are not relative to the project root
sys.path.insert(0, '/tracto/TrackToLearn') # Example path
sys.path.insert(0, '/tracto')            # Example path
# --- End Add project paths ---

# --- Imports from dependencies ---
try:
    from nibabel.streamlines.tractogram import Tractogram # LazyTractogram is less common now
    from dipy.io.streamline import load_tractogram, save_tractogram
    from dipy.tracking.streamline import Streamlines
    # from dipy.tracking.metrics import length, mean_curvature # Not used in generation loop
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    # from dipy.io.utils import (get_reference_info, create_tractogram_header) # Handled by StatefulTractogram
    # from dipy.core.geometry import angle_between_vectors # Not used
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure 'dipy', 'nibabel', 'torch', 'numpy', 'tqdm' are installed.")
    sys.exit(1)

try:
    from src.models.model import get_model
    from src.utils.configs import DataDict, TrainingConfig
    from src.loss_3d import Loss3D, visualize_3d_streamlines
    from environments.env import BaseEnv
except ImportError as e:
    print(f"Error importing project-specific modules: {e}")
    print("Ensure the script is run from a location where 'src' and 'environments' are accessible,")
    print("or adjust the sys.path insertions above.")
    sys.exit(1)

# Use AMP if available
try:
    from torch.cuda.amp import autocast
    amp_available = True
except ImportError:
    print("Warning: torch.cuda.amp not available. Running without Automatic Mixed Precision.")
    # Define a dummy autocast context manager if AMP is not available
    class dummy_autocast:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
    autocast = dummy_autocast
    amp_available = False


def generate_condition_vector(point: np.ndarray, env: BaseEnv) -> np.ndarray:
    """
    Project‑specific function provided by user – do **not** edit its core logic.
    Generates the condition vector for a *single* point using the environment.
    """
    # Ensure point is in the expected shape for the environment's _format_state
    point = np.asarray(point).reshape(1, 1, 3)
    return env._format_state(point)[0]

class StreamlineGenerator:
    def __init__(self, cfg: TrainingConfig, device: str = "cuda:0", use_amp: bool = False, use_torchscript: bool = True):
        self.cfg = cfg
        self.name = cfg.name
        self.iteration = 0
        self.epoch = 0
        self.training = False
        self.output_dir = cfg.output_dir
        self.use_torchscript = use_torchscript
        self.use_amp = use_amp and amp_available # Enable AMP only if requested and available

        try:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            if device != "cpu" and not torch.cuda.is_available():
                warn(f"Requested device '{device}' but CUDA not available. Falling back to CPU.")
            if self.use_amp and self.device == torch.device("cpu"):
                warn("AMP is enabled but running on CPU. AMP has no effect on CPU.")
                self.use_amp = False

        except Exception as e:
            print(f"Error setting device: {e}. Defaulting to CPU.")
            self.device = torch.device("cpu")
            self.use_amp = False

        # Initialize model
        self.model = get_model(config=cfg.model, device=self.device) # Pass device hint

        # Setup loss function (optional for inference, but kept for consistency)
        try:
            self.loss_func = Loss3D(cfg=cfg.loss)
            self.loss_func = self.loss_func.to(self.device)
        except AttributeError:
            print("Warning: Loss function configuration not found or invalid. Skipping loss setup.")
            self.loss_func = None
        except Exception as e:
            print(f"Warning: Could not initialize loss function: {e}. Skipping loss setup.")
            self.loss_func = None

        self.env = None # Initialize environment later in generate_all_streamlines
        self.seed_img = None # Initialize later

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print(f'Loading model snapshot from "{snapshot}"...')
        try:
            state_dict = torch.load(snapshot, map_location='cpu')
            model_dict = state_dict['state_dict']
            self.model.to(self.device) # Move model to device BEFORE loading state dict AND scripting
            load_result = self.model.load_state_dict(model_dict, strict=False)

            if load_result.missing_keys: warn(f'Missing keys: {load_result.missing_keys}')
            if load_result.unexpected_keys: warn(f'Unexpected keys: {load_result.unexpected_keys}')
            print('Model state dictionary loaded successfully.')

            # --- Apply TorchScript after loading weights and moving to device ---
            if self.use_torchscript:
                print("Attempting to compile model with TorchScript...")
                try:
                    self.model.eval()
                    dummy_condition = torch.randn(args.batch_size, C_DIM, device=self.device)
                    dummy_input_dict = {DataDict.condition: dummy_condition}
                    self.model = torch.jit.trace(self.model, example_kwarg_inputs=dummy_input_dict, strict=False)
                    print("Model compiled successfully with TorchScript.")
                except Exception as e:
                    warn(f"TorchScript compilation failed: {e}. Falling back to standard eager mode model.")
                    self.use_torchscript = False # Disable flag if compilation fails
                    # Ensure model is still on the correct device and in eval mode
                    self.model.to(self.device)
                    self.model.eval()
            # --- End TorchScript ---

            return state_dict
        except FileNotFoundError:
            print(f"Error: Snapshot file not found at {snapshot}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading snapshot: {e}")
            sys.exit(1)


    def load_learning_parameters(self, state_dict):
        """ Load epoch/iteration info from the snapshot if available. """
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            print(f'Model snapshot was saved at epoch: {self.epoch}.')
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print(f'Model snapshot was saved at iteration: {self.iteration}.')

    def set_eval_mode(self):
        """Set the model to evaluation mode and disable gradients."""
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)
        print("Model set to evaluation mode. Gradients disabled.")

    def _ensure_model_on_device(self):
        """Ensure the model parameters are on the designated device."""
        # Check if model has a device attribute or check parameter devices
        model_device = None
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            # Model might have no parameters, or is not yet fully initialized
            pass

        if model_device is not None and model_device != self.device:
            print(f"Moving model from {model_device} to {self.device}.")
            self.model.to(self.device)
        elif model_device is None:
             # If model has no parameters or device cannot be determined, try moving it anyway
            print(f"Attempting to ensure model is on {self.device}.")
            self.model.to(self.device)


    def load_wm_mask(self, wm_loc):
        """Load white matter mask from NIFTI file."""
        print(f"Loading white matter mask from: {wm_loc}")
        try:
            wm_img = nib.load(wm_loc)
            wm_data = wm_img.get_fdata(dtype=np.float32) # Load as float32
            # Ensure boolean mask (handle potential non-zero values)
            wm_mask = wm_data > 0.5 # Thresholding, adjust if mask uses different values
            print(f"WM mask loaded with shape: {wm_mask.shape}")
            return wm_mask, wm_img.affine
        except FileNotFoundError:
            print(f"Error: White matter mask file not found at {wm_loc}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading WM mask: {e}")
            sys.exit(1)

    def check_termination_conditions_vectorized(self, streamlines_batch, wm_mask, inv_wm_affine, max_length):
        """
        Check termination conditions for a batch of streamlines.

        Args:
            streamlines_batch (list): List of numpy arrays, each representing a streamline.
            wm_mask (np.ndarray): The white matter mask (boolean).
            inv_wm_affine (np.ndarray): Inverse of the WM mask affine transformation.
            max_length (int): Maximum allowed number of points per streamline.

        Returns:
            np.ndarray: A boolean array indicating if each streamline should terminate.
        """
        num_streamlines = len(streamlines_batch)
        terminate = np.zeros(num_streamlines, dtype=bool)
        last_points = np.zeros((num_streamlines, 3))
        valid_indices = [] # Indices of streamlines with at least one point

        for i, sl in enumerate(streamlines_batch):
            if len(sl) > 0:
                last_points[i] = sl[-1]
                valid_indices.append(i)
            else: # Should not happen if initialized correctly, but handle defensively
                terminate[i] = True

        if not valid_indices: # All streamlines were empty
             return terminate

        valid_indices = np.array(valid_indices)
        valid_last_points = last_points[valid_indices]

        # Check length
        lengths = np.array([len(sl) for sl in streamlines_batch])[valid_indices]
        terminate[valid_indices[lengths >= max_length]] = True

        # --- Boundary check (vectorized) ---
        # Convert last points to voxel coordinates
        # Add homogeneous coordinate (1)
        homogeneous_coords = np.hstack((valid_last_points, np.ones((len(valid_indices), 1))))
        # Apply inverse affine transformation
        voxel_coords_float = inv_wm_affine @ homogeneous_coords.T
        # Get first 3 coordinates and round to nearest integer
        voxel_coords = np.round(voxel_coords_float[:3, :]).astype(int).T

        # Get mask dimensions
        mask_shape = np.array(wm_mask.shape)

        # Check bounds (vectorized)
        out_of_bounds = np.any((voxel_coords < 0) | (voxel_coords >= mask_shape), axis=1)
        terminate[valid_indices[out_of_bounds]] = True

        # For points within bounds, check the WM mask value
        in_bounds_indices = valid_indices[~out_of_bounds]
        in_bounds_voxel_coords = voxel_coords[~out_of_bounds]

        if len(in_bounds_voxel_coords) > 0:
            # Use advanced indexing to get mask values efficiently
            mask_values = wm_mask[in_bounds_voxel_coords[:, 0],
                                  in_bounds_voxel_coords[:, 1],
                                  in_bounds_voxel_coords[:, 2]]
            # Terminate if mask value is False (or 0)
            terminate[in_bounds_indices[~mask_values]] = True

        return terminate


    def generate_streamlines_batch(self, seed_points_batch, subject_id, bundle, dataset_file, wm_mask, wm_affine, max_steps=200, max_segment_points=1):
        """
        Generate a batch of streamlines starting from seed points using batched inference.

        Args:
            seed_points_batch (np.ndarray): Array of seed points (N_batch, 3).
            subject_id (str): Subject identifier.
            bundle (str): Bundle name.
            dataset_file (str): Path to the dataset HDF5 file.
            wm_mask (np.ndarray): Boolean white matter mask.
            wm_affine (np.ndarray): Affine transformation for the WM mask.
            max_steps (int): Maximum number of segments to generate per streamline.
                             Total max length = max_steps * points_per_segment + 1 (seed)
            max_segment_points (int): Expected number of points per segment from the model.

        Returns:
            list: A list of generated streamlines (each streamline is a numpy array).
        """
        batch_size = len(seed_points_batch)
        inv_wm_affine = np.linalg.inv(wm_affine)
        max_length_points = max_steps * max_segment_points + 1 # +1 for the seed point

        # Initialize active streamlines: list of lists, each starting with a seed point
        active_streamlines = [[seed.copy()] for seed in seed_points_batch] # Use lists to append easily
        # Track which streamlines are still active (boolean array)
        is_active = np.ones(batch_size, dtype=bool)
        # Store completed streamlines
        completed_streamlines = [None] * batch_size

        # Main generation loop: continues as long as any streamline in the batch is active
        pbar = tqdm(total=max_steps, desc=f"Batch {bundle}/{subject_id}", leave=False)
        step = 0
        while np.any(is_active) and step < max_steps:
            active_indices = np.where(is_active)[0]
            current_batch_size = len(active_indices)

            # 1. Get the last point of each currently active streamline
            last_points = np.array([active_streamlines[i][-1] for i in active_indices])

            # 2. Generate condition vectors for the active points
            #    This part still iterates, but could be parallelized if env._format_state is slow
            #    and thread-safe or if it can be vectorized.
            condition_vectors = np.array([
                generate_condition_vector(point=last_points[j], env=self.env)
                for j in range(current_batch_size)
            ])

            # 3. Prepare batch for the model
            condition_tensor = torch.from_numpy(condition_vectors).float().to(self.device)
            input_dict = {DataDict.condition: condition_tensor}

            # 4. Perform batched inference with AMP context
            with torch.no_grad(), autocast(enabled=self.use_amp):
                output = self.model(input_dict=input_dict, sample=True) # Assuming model handles batch > 1
                # Ensure output is on CPU for numpy conversion
                new_segments_batch = output[DataDict.prediction].cpu().numpy() # Shape: (current_batch_size, N_points_segment, 3)

            # 5. Append new segments to active streamlines
            #    This loop iterates through the *active* streamlines only
            for i, active_idx in enumerate(active_indices):
                # Check if the corresponding streamline is still active before appending
                if is_active[active_idx]:
                    new_segment = new_segments_batch[i]
                    active_streamlines[active_idx].extend(new_segment) # Appends points from the segment

            # 6. Check termination conditions for all *currently active* streamlines
            #    Pass only the active streamlines to the vectorized check
            streamlines_to_check = [active_streamlines[i] for i in active_indices]
            terminate_flags = self.check_termination_conditions_vectorized(
                streamlines_to_check, wm_mask, inv_wm_affine, max_length_points
            )

            # 7. Update active status and store completed streamlines
            terminated_indices_in_batch = np.where(terminate_flags)[0]
            for idx_in_batch in terminated_indices_in_batch:
                original_idx = active_indices[idx_in_batch]
                if is_active[original_idx]: # Double check it hasn't been marked inactive already
                    is_active[original_idx] = False
                    # Convert list of points to numpy array before storing
                    completed_streamlines[original_idx] = np.array(active_streamlines[original_idx], dtype=np.float32)
                    # Optional: Clear the list in active_streamlines to save memory if needed
                    # active_streamlines[original_idx] = None # Or []

            pbar.update(1)
            step += 1

        pbar.close()

        # After the loop, handle any streamlines that were still active (reached max_steps)
        remaining_active_indices = np.where(is_active)[0]
        for idx in remaining_active_indices:
            completed_streamlines[idx] = np.array(active_streamlines[idx], dtype=np.float32)

        # Filter out any potential None entries if initialization failed (shouldn't happen)
        final_streamlines = [sl for sl in completed_streamlines if sl is not None]

        return final_streamlines


    def generate_all_streamlines(self, args):
        """Generate all streamlines using batched processing and save results."""
        start_time_total = time.time()

        # --- Setup ---
        print("\n--- Initializing Streamline Generation ---")
        os.makedirs(args.output_dir, exist_ok=True)
        output_trk_dir = os.path.dirname(args.output_trk)
        os.makedirs(output_trk_dir, exist_ok=True)

        # Ensure output_trk has .trk extension
        if not args.output_trk.lower().endswith('.trk'):
            base, _ = os.path.splitext(os.path.basename(args.output_trk))
            args.output_trk = os.path.join(output_trk_dir, f"{base}.trk")
            print(f"Adjusted output TRK path to: {args.output_trk}")

        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Load WM mask once
        wm_mask, wm_affine = self.load_wm_mask(args.wm_loc)

        # Load model snapshot
        print(f"\nLoading model from {args.model_path}...")
        state_dict = self.load_snapshot(args.model_path)
        self.load_learning_parameters(state_dict)
        self.set_eval_mode()
        self._ensure_model_on_device() # Ensure model is on the correct device after loading

        # Load reference image for saving tractogram
        try:
            self.seed_img = nib.load(args.wm_loc) # Use WM mask NIFTI as reference
            print(f"Using '{args.wm_loc}' as reference NIFTI for saving.")
        except FileNotFoundError:
            print(f"Error: Reference NIFTI file not found at {args.wm_loc}")
            sys.exit(1)
        except Exception as e:
             print(f"Error loading reference NIFTI: {e}")
             sys.exit(1)


        # Initialize environment (needs access to data)
        print("\nInitializing environment...")
        try:
            # Simplified env parameters based on original script, adjust if needed
            self.env = BaseEnv(
                dataset_file=args.dataset_file,
                wm_loc=args.wm_loc,        # Provide WM location to env if needed
                subject_id=args.subject,   # Env might use subject ID
                step_size=0.2,             # Example value, adjust if critical
                max_length=args.max_steps, # Use max_steps from args
                n_signal=1, n_dirs=8, max_angle=60, min_length=10,
                n_seeds_per_voxel=1, # Set to 1 as we provide seeds explicitly
                rng=np.random.RandomState(1337),
                add_neighborhood=1.5,
                compute_reward=False, # No reward needed for generation
                device="cpu", # Env operations likely on CPU, model is on self.device
            )
            print("Environment initialized.")
        except Exception as e:
            print(f"Error initializing environment: {e}")
            sys.exit(1)


        # Load seed streamlines
        print(f"\nLoading seed streamlines from {args.seed_trk}...")
        try:
            # Load into RASMM space, relative to the reference NIFTI's world space
            seed_tractogram = load_tractogram(args.seed_trk, self.seed_img,
                                              to_space=Space.RASMM, bbox_valid_check=False)
            seed_streamlines_all = seed_tractogram.streamlines
            num_total_seeds = len(seed_streamlines_all)
            # Limit seeds if requested
            num_seeds_to_process = min(num_total_seeds, args.num_streamlines) if args.num_streamlines > 0 else num_total_seeds

            if num_seeds_to_process == 0:
                 print("Error: No seed streamlines found or requested.")
                 sys.exit(1)

            print(f"Loaded {num_total_seeds} seed streamlines. Processing {num_seeds_to_process}.")

            # Extract seed points (first point of each streamline)
            seed_points = np.array([s[0] for s in seed_streamlines_all[:num_seeds_to_process]], dtype=np.float32)

        except FileNotFoundError:
            print(f"Error: Seed TRK file not found at {args.seed_trk}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading seed streamlines: {e}")
            sys.exit(1)

        # --- Batch Generation ---
        print(f"\n--- Starting Streamline Generation (Batch Size: {args.batch_size}) ---")
        generated_streamlines_all = []
        batch_times = []

        max_model_steps = args.max_steps # Max segments model should generate

        # Process seeds in batches
        for i in tqdm(range(0, num_seeds_to_process, args.batch_size), desc="Processing Batches"):
            batch_start_time = time.time()
            start_idx = i
            end_idx = min(i + args.batch_size, num_seeds_to_process)
            current_batch_seed_points = seed_points[start_idx:end_idx]

            if len(current_batch_seed_points) == 0:
                continue # Should not happen with proper range calculation

            # Generate streamlines for the current batch
            generated_batch = self.generate_streamlines_batch(
                seed_points_batch=current_batch_seed_points,
                subject_id=args.subject,
                bundle=args.bundle,
                dataset_file=args.dataset_file,
                wm_mask=wm_mask,
                wm_affine=wm_affine,
                max_steps=max_model_steps,
                max_segment_points=self.cfg.model.get('points_per_segment', 1) # Get expected segment length from config if possible
            )

            generated_streamlines_all.extend(generated_batch)
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            # Optional: print batch time here if needed

        total_generation_time = time.time() - start_time_total
        print(f"\n--- Streamline Generation Finished ---")
        print(f"Successfully generated {len(generated_streamlines_all)} streamlines.")

        # --- Saving Results ---
        if not generated_streamlines_all:
            print("Warning: No streamlines were generated. Skipping saving and visualization.")
            return

        print(f"\nSaving {len(generated_streamlines_all)} generated streamlines to {args.output_trk}")
        try:
            # Create Stateful Tractogram: requires streamlines, reference image, and space
            sft = StatefulTractogram(
                Streamlines(generated_streamlines_all), # Pass the list of numpy arrays
                self.seed_img,                          # Reference NIFTI (e.g., WM mask img)
                Space.RASMM)                            # Space matches seed loading space

            # Save using dipy's save_tractogram
            save_tractogram(sft, args.output_trk, bbox_valid_check=False)
            print("Tractogram saved successfully.")
        except Exception as e:
            print(f"Error saving tractogram: {e}")


        # --- Final Visualizations ---
        print("\nGenerating final visualizations...")
        try:
            # Use a subset for visualization if too many streamlines were generated
            num_vis = min(len(generated_streamlines_all), 500) # Limit visualization complexity
            indices_vis = np.random.choice(len(generated_streamlines_all), num_vis, replace=False)
            generated_vis = [generated_streamlines_all[i] for i in indices_vis]

            # Prepare ground truth (seeds) for comparison - use the same number as generated_vis
            if len(seed_streamlines_all) >= num_vis:
                 # Select corresponding seed streamlines or a random subset if indices don't match
                 gt_vis_indices = np.random.choice(len(seed_streamlines_all), num_vis, replace=False)
                 gt_vis = [seed_streamlines_all[i] for i in gt_vis_indices]
            else: # Less seeds than generated? Use all seeds
                 gt_vis = list(seed_streamlines_all[:num_seeds_to_process]) # Use processed seeds


            # Convert lists of arrays to single large arrays for visualize_3d_streamlines if needed
            # Check the expected input format of visualize_3d_streamlines
            # Assuming it takes a list of np.arrays or a single stacked np.array
            # If it needs stacked:
            # all_generated_vis = np.vstack([np.array(sl) for sl in generated_vis])
            # all_ground_truth_vis = np.vstack([np.array(sl) for sl in gt_vis])
            # If it takes lists:
            all_generated_vis = generated_vis
            all_ground_truth_vis = gt_vis

            # 1. Plot of only predicted streamlines
            pred_vis_file = os.path.join(vis_dir, "predicted_streamlines_vis.png")
            print(f"Saving predicted streamlines visualization ({num_vis} streamlines) to {pred_vis_file}")
            visualize_3d_streamlines(
                predictions=all_generated_vis,
                ground_truth=all_generated_vis, # Set ground_truth to None or same as predictions
                subject_id=args.subject,
                bundle=args.bundle,
                split="testset",
                output_file=pred_vis_file
            )

            # 2. Comparison plot of GT (seeds) and generated streamlines
            comp_vis_file = os.path.join(vis_dir, "comparison_streamlines_vis.png")
            print(f"Saving comparison visualization ({num_vis} GT vs {num_vis} Pred) to {comp_vis_file}")
            visualize_3d_streamlines(
                predictions=all_generated_vis,
                ground_truth=all_ground_truth_vis,
                subject_id=args.subject,
                bundle=args.bundle,
                split="testset", 
                output_file=comp_vis_file
            )
            print("Visualizations saved.")

        except Exception as e:
            print(f"Warning: Could not generate visualizations. Error: {e}")


        # --- Timing Summary ---
        print("\n=== Performance Summary ===")
        print(f"Total execution time: {total_generation_time:.2f} seconds")
        if batch_times:
            print(f"Number of batches processed: {len(batch_times)}")
            print(f"Average time per batch: {np.mean(batch_times):.2f} seconds")
            print(f"Fastest batch: {min(batch_times):.2f} seconds")
            print(f"Slowest batch: {max(batch_times):.2f} seconds")
            print(f"Std dev batch time: {np.std(batch_times):.2f} seconds")
        avg_time_per_streamline = total_generation_time / len(generated_streamlines_all) if generated_streamlines_all else 0
        print(f"Approx. average time per streamline: {avg_time_per_streamline:.4f} seconds")
        print(f"GPU used: {self.device}")
        print(f"Automatic Mixed Precision (AMP) used: {self.use_amp}")
        print("==========================")

        print("\nScript finished!")
        print(f"Generated TRK file: {args.output_trk}")
        print(f"Visualizations saved in: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate streamlines using a diffusion model with batch processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values
    )

    # Required arguments
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--bundle", type=str, required=True, help="Bundle name")
    parser.add_argument("--seed_trk", type=str, required=True, help="Path to seed streamline file (.trk)")
    parser.add_argument("--output_trk", type=str, required=True, help="Path to output streamline file (.trk)")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to HDF5 dataset file used by the environment")
    parser.add_argument("--wm_loc", type=str, required=True, help="Path to white matter mask NIFTI file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pth or similar)")

    # Optimization arguments
    parser.add_argument("--batch_size", type=int, default=128, help="Number of streamlines to process in parallel per batch")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--use_amp", action='store_true', help="Enable Automatic Mixed Precision (AMP) for potentially faster inference on compatible GPUs")

    # Generation control arguments
    parser.add_argument("--num_streamlines", type=int, default=-1, help="Maximum number of streamlines to generate (-1 to use all seeds from seed_trk)")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of generation steps (segments) per streamline")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output_generated", help="Directory to save visualizations and potentially other outputs")

    args = parser.parse_args()

    cfg = TrainingConfig
    cfg.name = "TractoDiff_Generator"
    cfg.model.type = "diffusion"
    cfg.model.generator_type = "diffusion"

    # Initialize generator and run
    generator = StreamlineGenerator(cfg, device=args.device, use_amp=args.use_amp)
    generator.generate_all_streamlines(args)

if __name__ == "__main__":
    main()