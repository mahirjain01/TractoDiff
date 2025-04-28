#!/usr/bin/env python

import argparse
import numpy as np
import nibabel as nib
import os
import sys
sys.path.insert(0, '/tracto/TrackToLearn')
sys.path.insert(0, '/tracto')

from environments.env import BaseEnv

def generate_condition_vector(
    point: np.ndarray,
    subject_id: str,
    dataset_file: str,
    wm_loc: str,
    n_signal: int = 1,
    n_dirs: int = 8,
    step_size: float = 0.2,
    max_angle: float = 60,
    min_length: float = 10,
    max_length: float = 200,
    n_seeds_per_voxel: int = 4,
    add_neighborhood: float = 1.5,
    device = None
) -> np.ndarray:
    """
    Generate a condition vector for a given 3D point in voxel space.
    
    Parameters
    ----------
    point : np.ndarray
        3D coordinate in voxel space [x, y, z]
    subject_id : str
        Subject ID for loading correct MRI data
    dataset_file : str
        Path to HDF5 file containing diffusion data
    wm_loc : str
        Path to white matter mask
    n_signal : int
        Number of signal "history" to keep in input
    n_dirs : int
        Number of last actions to append to input
    step_size : float
        Step size for tracking
    max_angle : float
        Maximum angle for tracking
    min_length : int
        Minimum length for streamlines
    max_length : int
        Maximum length for streamlines
    n_seeds_per_voxel : int
        How many seeds to generate per voxel
    add_neighborhood : float
        Use signal in neighboring voxels for model input
    device : torch.device
        Device to run on (CPU/GPU)
        
    Returns
    -------
    np.ndarray
        Condition vector of shape (346,) for the input point
    """
    
    # Initialize environment with same parameters as original script
    env = BaseEnv(
        dataset_file=dataset_file,
        wm_loc=wm_loc,
        subject_id=subject_id,
        n_signal=n_signal,
        n_dirs=n_dirs,
        step_size=step_size,
        max_angle=max_angle,
        min_length=min_length,
        max_length=max_length,
        n_seeds_per_voxel=n_seeds_per_voxel,
        rng=np.random.RandomState(seed=1337),
        add_neighborhood=add_neighborhood,
        compute_reward=True,
        device=device
    )
    
    # Format point into expected shape (1, 1, 3)
    point = np.array(point).reshape(1, 1, 3)
    
    # Generate condition vector using _format_state
    condition_vector = env._format_state(point)
    
    # Return first (and only) condition vector
    return condition_vector[0]

def main():
    parser = argparse.ArgumentParser(description="Generate condition vector for a 3D point")
    
    # Required arguments
    parser.add_argument("point", type=float, nargs=3, help="3D point coordinates in voxel space [x y z]")
    parser.add_argument("subject_id", type=str, help="Subject ID")
    parser.add_argument("dataset_file", type=str, help="Path to HDF5 dataset file")
    parser.add_argument("wm_loc", type=str, help="Path to white matter mask")
    
    # Optional arguments with defaults matching original script
    parser.add_argument("--n_signal", type=int, default=1, help="Number of signal history")
    parser.add_argument("--n_dirs", type=int, default=8, help="Number of last actions")
    parser.add_argument("--step_size", type=float, default=0.2, help="Step size")
    parser.add_argument("--max_angle", type=float, default=60, help="Maximum angle")
    parser.add_argument("--min_length", type=float, default=10, help="Minimum length")
    parser.add_argument("--max_length", type=float, default=200, help="Maximum length")
    parser.add_argument("--n_seeds_per_voxel", type=int, default=4, help="Seeds per voxel")
    parser.add_argument("--add_neighborhood", type=float, default=1.5, help="Neighborhood size")
    
    args = parser.parse_args()
    
    # Generate condition vector
    condition_vector = generate_condition_vector(
        point=args.point,
        subject_id=args.subject_id,
        dataset_file=args.dataset_file,
        wm_loc=args.wm_loc,
        n_signal=args.n_signal,
        n_dirs=args.n_dirs,
        step_size=args.step_size,
        max_angle=args.max_angle,
        min_length=args.min_length,
        max_length=args.max_length,
        n_seeds_per_voxel=args.n_seeds_per_voxel,
        add_neighborhood=args.add_neighborhood
    )
    
    # Print shape and first few values
    print(f"Condition vector shape: {condition_vector.shape}")
    print(f"First 10 values: {condition_vector[:10]}")
    
    return condition_vector

if __name__ == "__main__":
    main() 