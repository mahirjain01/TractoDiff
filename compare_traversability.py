#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shutil
import argparse
import json
import glob
import torch
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append('.')

# We need to patch the configs before they are imported
def patch_configs():
    """Patch the configs module to read traversability from environment variable"""
    # First, try to monkey patch the config module
    try:
        from src.utils.configs import LossConfig
        if "USE_TRAVERSABILITY" in os.environ:
            traversability_value = os.environ["USE_TRAVERSABILITY"].lower() in ("1", "true", "yes", "on")
            print(f"Setting traversability to {traversability_value} from environment variable")
            LossConfig.use_traversability = traversability_value
    except ImportError:
        print("Warning: Could not import and patch configs directly")

# Apply the patch
patch_configs()

from src.utils.configs import LossConfig, TrainingConfig
import nibabel as nib

def setup_args():
    parser = argparse.ArgumentParser(description="Compare traversability ON vs OFF in TractoDiff")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="traversability_comparison", help="Output directory for visualizations")
    parser.add_argument("--subject_id", type=str, default=None, help="Subject ID to process")
    parser.add_argument("--existing_results", action="store_true", help="Use existing results if available")
    return parser.parse_args()

def save_config(output_dir, status, config_dict):
    """Save configuration to a JSON file"""
    config_file = os.path.join(output_dir, f"traversability_{status}_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    return config_file

def modify_configs_file(use_traversability):
    """Create a modified configs file that sets traversability"""
    configs_file = "src/utils/configs.py"
    
    # Read the original config file
    with open(configs_file, 'r') as f:
        content = f.read()
    
    # Create a backup if it doesn't exist
    backup_file = configs_file + ".backup"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"Created backup of original config at {backup_file}")
    
    # Replace the use_traversability line
    if "LossConfig.use_traversability = " in content:
        new_content = content.replace(
            "LossConfig.use_traversability = True", 
            f"LossConfig.use_traversability = {str(use_traversability)}"
        )
        new_content = new_content.replace(
            "LossConfig.use_traversability = False", 
            f"LossConfig.use_traversability = {str(use_traversability)}"
        )
        
        # Write the modified file
        with open(configs_file, 'w') as f:
            f.write(new_content)
        
        print(f"Modified {configs_file} to set use_traversability={use_traversability}")
        return True
    else:
        print(f"Warning: Could not find traversability setting in {configs_file}")
        return False

def restore_configs_file():
    """Restore the original configs file from backup"""
    configs_file = "src/utils/configs.py"
    backup_file = configs_file + ".backup"
    
    if os.path.exists(backup_file):
        shutil.copy(backup_file, configs_file)
        print(f"Restored original config from {backup_file}")
        return True
    
    return False

def run_with_traversability(use_traversability, model_path, output_dir, subject_id=None):
    """Run inference with traversability enabled or disabled"""
    print(f"Running inference with traversability {'ENABLED' if use_traversability else 'DISABLED'}")
    
    # Create output directory
    status = "on" if use_traversability else "off"
    output_subdir = os.path.join(output_dir, f"traversability_{status}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Create a modified config file for running inference
    config_dict = {
        "use_traversability": use_traversability,
        "model_path": model_path,
        "output_dir": output_subdir,
        "subject_id": subject_id
    }
    
    # Save config for reference
    config_file = save_config(output_dir, status, config_dict)
    
    # Modify the configs file directly (more reliable than environment variables)
    modified = modify_configs_file(use_traversability)
    
    try:
        # Run inference using the current scripts available
        cmd = [
            "python", "inference_cuda_optimized.py",
            "--snapshot", model_path,
            "--output_dir", output_subdir,
        ]
        
        if subject_id:
            cmd.extend(["--subject", subject_id])
        
        # Set traversability flag in the environment for the subprocess
        env = os.environ.copy()
        env["USE_TRAVERSABILITY"] = "1" if use_traversability else "0"
        
        print(f"Running command: {' '.join(cmd)}")
        os.environ["USE_TRAVERSABILITY"] = "1" if use_traversability else "0"
        
        # Modify the global config before running
        orig_value = LossConfig.use_traversability
        try:
            # Set the config value
            LossConfig.use_traversability = use_traversability
            
            # Run the command
            import subprocess
            result = subprocess.run(cmd, env=env, check=True)
            print(f"Command completed with exit code {result.returncode}")
            
            return output_subdir
        finally:
            # Restore original config value
            LossConfig.use_traversability = orig_value
    finally:
        # Restore original configs file if we modified it
        if modified:
            restore_configs_file()

def collect_images(directory):
    """Collect all images from the directory"""
    image_files = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_files.extend(glob.glob(os.path.join(directory, f"**/*.{ext}"), recursive=True))
    return sorted(image_files)

def create_comparison_visualizations(trav_on_dir, trav_off_dir, output_dir):
    """Create side-by-side comparisons of images from both runs"""
    on_images = collect_images(trav_on_dir)
    off_images = collect_images(trav_off_dir)
    
    print(f"Found {len(on_images)} images with traversability ON")
    print(f"Found {len(off_images)} images with traversability OFF")
    
    # Match images by filename
    on_basenames = {os.path.basename(p): p for p in on_images}
    off_basenames = {os.path.basename(p): p for p in off_images}
    
    # Find common images
    common_basenames = set(on_basenames.keys()).intersection(set(off_basenames.keys()))
    print(f"Found {len(common_basenames)} matching images to compare")
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate comparisons
    for basename in common_basenames:
        on_path = on_basenames[basename]
        off_path = off_basenames[basename]
        
        # Load images
        on_img = plt.imread(on_path)
        off_img = plt.imread(off_path)
        
        # Create figure for comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot traversability ON
        axes[0].imshow(on_img)
        axes[0].set_title(f"Traversability ON")
        axes[0].axis('off')
        
        # Plot traversability OFF
        axes[1].imshow(off_img)
        axes[1].set_title(f"Traversability OFF")
        axes[1].axis('off')
        
        # Calculate difference image and visualize
        diff_img = np.abs(on_img - off_img)
        if diff_img.ndim == 3 and diff_img.shape[2] >= 3:
            # Convert to grayscale for difference if RGB
            diff_gray = np.mean(diff_img[:,:,:3], axis=2)
            # Normalize for better visibility
            diff_normalized = diff_gray / np.max(diff_gray) if np.max(diff_gray) > 0 else diff_gray
            # Create a heatmap from black (no diff) to red (max diff)
            cmap = LinearSegmentedColormap.from_list("diff_cmap", [(0, 0, 0, 0), (1, 0, 0, 1)])
            axes[2].imshow(on_img)  # Background image for context
            axes[2].imshow(diff_normalized, cmap=cmap, alpha=0.7)  # Overlay difference
        else:
            # Fallback for non-RGB images
            axes[2].imshow(diff_img, cmap='hot')
            
        axes[2].set_title(f"Differences (red highlights)")
        axes[2].axis('off')
        
        # Save the comparison
        out_path = os.path.join(comparison_dir, f"compare_{basename}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        
        print(f"Created comparison for {basename}")
    
    # Create a summary visualization showing the most significant differences
    create_summary_visualization(common_basenames, on_basenames, off_basenames, comparison_dir)
    
    return comparison_dir

def create_summary_visualization(common_basenames, on_basenames, off_basenames, comparison_dir):
    """Create a summary visualization showing the most significant differences"""
    # Calculate difference metrics for each image pair
    diff_metrics = []
    
    for basename in common_basenames:
        on_path = on_basenames[basename]
        off_path = off_basenames[basename]
        
        on_img = plt.imread(on_path)
        off_img = plt.imread(off_path)
        
        # Calculate mean absolute difference
        diff = np.abs(on_img - off_img)
        if diff.ndim == 3 and diff.shape[2] >= 3:
            # Use mean of RGB channels for color images
            diff_mean = np.mean(diff[:,:,:3])
        else:
            diff_mean = np.mean(diff)
            
        diff_metrics.append((basename, diff_mean))
    
    # Sort by difference (highest first)
    diff_metrics.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 5 differences for summary (or fewer if less available)
    top_diffs = diff_metrics[:min(5, len(diff_metrics))]
    
    if not top_diffs:
        print("No differences found for summary visualization")
        return
    
    # Create summary figure
    fig, axes = plt.subplots(len(top_diffs), 3, figsize=(18, 6*len(top_diffs)))
    
    # Handle case with only one row
    if len(top_diffs) == 1:
        axes = [axes]
    
    for i, (basename, diff_value) in enumerate(top_diffs):
        on_path = on_basenames[basename]
        off_path = off_basenames[basename]
        
        on_img = plt.imread(on_path)
        off_img = plt.imread(off_path)
        
        # Plot traversability ON
        axes[i][0].imshow(on_img)
        axes[i][0].set_title(f"Traversability ON")
        axes[i][0].axis('off')
        
        # Plot traversability OFF
        axes[i][1].imshow(off_img)
        axes[i][1].set_title(f"Traversability OFF")
        axes[i][1].axis('off')
        
        # Calculate difference image
        diff_img = np.abs(on_img - off_img)
        if diff_img.ndim == 3 and diff_img.shape[2] >= 3:
            diff_gray = np.mean(diff_img[:,:,:3], axis=2)
            norm_diff = diff_gray / np.max(diff_gray) if np.max(diff_gray) > 0 else diff_gray
            cmap = LinearSegmentedColormap.from_list("diff_cmap", [(0, 0, 0, 0), (1, 0, 0, 1)])
            axes[i][2].imshow(on_img)
            axes[i][2].imshow(norm_diff, cmap=cmap, alpha=0.7)
        else:
            axes[i][2].imshow(diff_img, cmap='hot')
            
        axes[i][2].set_title(f"Diff Score: {diff_value:.4f}")
        axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "summary_top_differences.png"), dpi=300)
    plt.close(fig)
    
    print(f"Created summary visualization of top {len(top_diffs)} differences")

def generate_report(output_dir, trav_on_dir, trav_off_dir, comparison_dir):
    """Generate an HTML report summarizing the findings"""
    report_path = os.path.join(output_dir, "traversability_report.html")
    
    # Find comparison images
    comparison_images = glob.glob(os.path.join(comparison_dir, "*.png"))
    comparison_images.sort()
    
    # Place summary image at the top if it exists
    summary_path = os.path.join(comparison_dir, "summary_top_differences.png")
    if os.path.exists(summary_path) and summary_path in comparison_images:
        comparison_images.remove(summary_path)
        comparison_images.insert(0, summary_path)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traversability Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .image-container {{ margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Traversability Comparison Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>This report compares the effect of the <code>use_traversability</code> flag on the generation of streamlines.</p>
            <ul>
                <li><strong>Traversability ON:</strong> {trav_on_dir}</li>
                <li><strong>Traversability OFF:</strong> {trav_off_dir}</li>
                <li><strong>Comparisons:</strong> {comparison_dir}</li>
                <li><strong>Total comparisons:</strong> {len(comparison_images)}</li>
            </ul>
            <p>In the comparison images:</p>
            <ul>
                <li>The <strong>left panel</strong> shows the image with traversability ON</li>
                <li>The <strong>middle panel</strong> shows the image with traversability OFF</li>
                <li>The <strong>right panel</strong> highlights the differences in red</li>
            </ul>
        </div>
        
        <h2>Image Comparisons</h2>
    """
    
    # Add each comparison image
    for img_path in comparison_images:
        rel_path = os.path.relpath(img_path, output_dir)
        basename = os.path.basename(img_path)
        html_content += f"""
        <div class="image-container">
            <h3>{basename}</h3>
            <img src="{rel_path}" alt="{basename}">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated report at {report_path}")
    return report_path

def main():
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for traversability ON and OFF runs
    trav_on_dir = os.path.join(args.output_dir, "traversability_on")
    trav_off_dir = os.path.join(args.output_dir, "traversability_off")
    
    # Run inference with both settings if needed
    if not args.existing_results or not os.path.exists(trav_on_dir):
        run_with_traversability(True, args.model_path, args.output_dir, args.subject_id)
    else:
        print(f"Using existing results with traversability ON from {trav_on_dir}")
        
    if not args.existing_results or not os.path.exists(trav_off_dir):
        run_with_traversability(False, args.model_path, args.output_dir, args.subject_id)
    else:
        print(f"Using existing results with traversability OFF from {trav_off_dir}")
    
    # Create comparisons between the two results
    comparison_dir = create_comparison_visualizations(trav_on_dir, trav_off_dir, args.output_dir)
    
    # Generate an HTML report
    report_path = generate_report(args.output_dir, trav_on_dir, trav_off_dir, comparison_dir)
    
    print("\nTraversability comparison complete!")
    print(f"Results are available at: {args.output_dir}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main() 