#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def visualize_trk_simple(trk_file, output_dir, reference_img=None, subset=2000):
    """
    Generate and save simple visualizations of a tractogram using matplotlib
    
    Parameters
    ----------
    trk_file : str
        Path to the TRK file
    output_dir : str
        Directory to save visualizations
    reference_img : str, optional
        Path to reference image
    subset : int, optional
        Number of streamlines to visualize
    """
    print(f"Loading tractogram: {trk_file}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference image if provided
    if reference_img:
        ref_img = nib.load(reference_img)
    else:
        ref_img = None
    
    # Load tractogram
    tractogram = load_tractogram(trk_file, ref_img, bbox_valid_check=False, to_space=Space.RASMM)
    streamlines = tractogram.streamlines
    
    print(f"Loaded {len(streamlines)} streamlines")
    
    # Take a subset for visualization
    if subset and subset < len(streamlines):
        print(f"Using subset of {subset} streamlines for visualization")
        indices = np.random.choice(len(streamlines), size=subset, replace=False)
        streamlines_subset = streamlines[indices]
    else:
        streamlines_subset = streamlines
    
    # Get all points for projections
    print("Extracting all points...")
    all_points = np.vstack([s for s in streamlines_subset])
    
    # Create 2D projections (top, front, side views)
    print("Creating 2D projections...")
    
    # Create 2D histograms (projections)
    plt.figure(figsize=(15, 4))
    
    plt.subplot(131)
    plt.hist2d(all_points[:, 0], all_points[:, 1], bins=100, cmap='hot')
    plt.title('Axial Projection (X-Y)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(132)
    plt.hist2d(all_points[:, 0], all_points[:, 2], bins=100, cmap='hot')
    plt.title('Sagittal Projection (X-Z)')
    plt.xlabel('X')
    plt.ylabel('Z')
    
    plt.subplot(133)
    plt.hist2d(all_points[:, 1], all_points[:, 2], bins=100, cmap='hot')
    plt.title('Coronal Projection (Y-Z)')
    plt.xlabel('Y')
    plt.ylabel('Z')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tractogram_projections.png'), dpi=300)
    plt.close()
    
    # Create a 3D plot of a subset of streamlines
    print("Creating 3D matplotlib visualization...")
    
    # Use fewer streamlines for 3D plot to avoid clutter
    viz_streamlines = streamlines_subset[:min(200, len(streamlines_subset))]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate streamline lengths for coloring
    streamline_lengths = np.array([len(s) for s in viz_streamlines])
    norm = Normalize(vmin=np.min(streamline_lengths), vmax=np.max(streamline_lengths))
    
    for i, streamline in enumerate(viz_streamlines):
        # Get normalized color based on streamline length
        color_val = plt.cm.plasma(norm(streamline_lengths[i]))
        
        # Plot the streamline
        ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], 
                linewidth=1, alpha=0.5, c=color_val)
    
    # Set equal aspect ratio for all axes
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) / 2
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) / 2
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Tractography 3D Visualization')
    
    # Save from different views
    for angle_elevation in [20, 0, -20]:
        for angle_azimuth in [0, 45, 90, 135, 180, 225, 270, 315]:
            ax.view_init(elev=angle_elevation, azim=angle_azimuth)
            plt.savefig(os.path.join(output_dir, f'3d_view_elev{angle_elevation}_azim{angle_azimuth}.png'), 
                        dpi=200, bbox_inches='tight')
    
    # Create length distribution histogram
    plt.figure(figsize=(10, 6))
    all_lengths = np.array([len(s) for s in streamlines])
    plt.hist(all_lengths, bins=50)
    plt.xlabel('Streamline Length (number of points)')
    plt.ylabel('Count')
    plt.title('Distribution of Streamline Lengths')
    plt.savefig(os.path.join(output_dir, 'length_distribution.png'), dpi=300)
    plt.close()
    
    # Create a bundle visualization from multiple angles
    print("Creating bundle visualization from different angles...")
    
    # Create 3 different view angles
    views = [
        {'elev': 20, 'azim': 30, 'name': 'perspective'},
        {'elev': 90, 'azim': 0, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'front'},
        {'elev': 0, 'azim': 90, 'name': 'side'}
    ]
    
    for view in views:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use all streamlines but with higher transparency
        for streamline in viz_streamlines:
            ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], 
                    linewidth=0.5, alpha=0.2, color='blue')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Bundle View - {view["name"]}')
        
        # Set view angle
        ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Set equal aspect ratio
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.savefig(os.path.join(output_dir, f'bundle_view_{view["name"]}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All visualizations saved to: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize TRK files using matplotlib")
    parser.add_argument("trk_file", help="Path to input TRK file")
    parser.add_argument("--output_dir", default="trk_visualizations", help="Directory to save visualizations")
    parser.add_argument("--reference", help="Path to reference NIFTI file")
    parser.add_argument("--subset", type=int, default=2000, help="Number of streamlines to visualize (default: 2000)")
    
    args = parser.parse_args()
    
    visualize_trk_simple(
        trk_file=args.trk_file,
        output_dir=args.output_dir,
        reference_img=args.reference,
        subset=args.subset
    )

if __name__ == "__main__":
    main()