#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import nibabel as nib
import time

from nibabel.streamlines.tractogram import LazyTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.tracking.metrics import length, mean_curvature
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info, create_tractogram_header
from warnings import warn

# Add project root to path if needed
import os, sys
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, '/tracto/TrackToLearn')
sys.path.insert(0, '/tracto')



from src.models.model import get_model
from src.utils.configs import DataDict, TrainingConfig
from src.loss_3d import Loss3D, visualize_3d_streamlines
from environments.env import BaseEnv

class StreamlineGenerator:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.iteration = 0
        self.epoch = 0
        self.training = False
        self.output_dir = cfg.output_dir

        # Enable cuDNN benchmark for optimized kernels
        cudnn.benchmark = True

        # Device setup
        self.device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cuda:0')
        print(f"Using device: {self.device}")

        # Initialize model and wrap for multi-GPU if available
        self.model = get_model(config=cfg.model, device=self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

        # AMP scaler for inference (autocast only)
        # No scaler needed since no gradients

        # Timing
        self.streamline_times = []

    def load_snapshot(self, snapshot):
        print(f'Loading checkpoint from {snapshot}')
        state = torch.load(snapshot, map_location=self.device)
        model_dict = state['state_dict']
        self.model.load_state_dict(model_dict, strict=False)
        self.epoch = state.get('epoch', 0)
        self.iteration = state.get('iteration', 0)
        print(f"Loaded model at epoch {self.epoch}, iteration {self.iteration}")

    def load_wm_mask(self, wm_loc):
        img = nib.load(wm_loc)
        return img.get_fdata(), img.affine

    def check_termination_conditions(self, streamline: np.ndarray, mask: np.ndarray, affine: np.ndarray,
                                     max_length=200):
        if len(streamline) >= max_length:
            return True
        # Boundary check
        last_point = streamline[-1]
        vox = np.linalg.inv(affine).dot(np.append(last_point, 1))[:3]
        vox = np.round(vox).astype(int)
        x, y, z = vox
        if x<0 or y<0 or z<0 or x>=mask.shape[0] or y>=mask.shape[1] or z>=mask.shape[2]:
            return True
        if mask[x, y, z] == 0:
            return True
        return False

    def generate_streamlines_batch(self, seed_points: list, wm_mask, wm_affine):
        """
        Generate a batch of streamlines in parallel until termination.
        """
        num = len(seed_points)
        # Initialize per-streamline data
        streamlines = [[pt] for pt in seed_points]
        current_pts = np.array([pt for pt in seed_points])
        active = np.ones(num, dtype=bool)

        while active.any():
            # Collect condition vectors for active streamlines
            conds = [
                BaseEnv._format_state(  # use static format_state from env
                    np.asarray(current_pts[i]).reshape(1,1,3)
                )[0]
                for i in np.where(active)[0]
            ]
            conds = torch.tensor(np.stack(conds), dtype=torch.float32, device=self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                out = self.model({DataDict.condition: conds}, sample=True)
                preds = out[DataDict.prediction].cpu().numpy()

            # Update each active streamline
            for idx, seg in zip(np.where(active)[0], preds):
                streamlines[idx].extend(seg)
                current_pts[idx] = seg[-1]
                if self.check_termination_conditions(np.array(streamlines[idx]), wm_mask, wm_affine):
                    active[idx] = False

        return [np.array(sl) for sl in streamlines]

    def generate_all_streamlines(self, args):
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.output_trk.endswith('.trk'):
            base = os.path.splitext(args.output_trk)[0]
            args.output_trk = base + '.trk'

        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        self.load_snapshot(args.model_path)

        # Load environment and WM mask
        self.env = BaseEnv(
            dataset_file=args.dataset_file,
            wm_loc=args.wm_loc,
            subject_id=args.subject,
            n_signal=1, n_dirs=8,
            step_size=0.2, max_angle=60,
            min_length=10, max_length=200,
            n_seeds_per_voxel=4,
            rng=np.random.RandomState(1337),
            add_neighborhood=1.5,
            compute_reward=True,
            device=self.device,
        )
        wm_mask, wm_affine = self.load_wm_mask(args.wm_loc)

        seed_tract = load_tractogram(args.seed_trk, reference='same', bbox_valid_check=False, to_space=Space.RASMM)
        seeds = list(seed_tract.streamlines)
        seed_points = [sl[0] for sl in seeds]
        print(f"Generating {len(seed_points)} streamlines in batches of {args.batch_size}")

        all_streamlines = []
        for i in range(0, len(seed_points), args.batch_size):
            batch_seeds = seed_points[i:i+args.batch_size]
            start = time.time()
            batch_out = self.generate_streamlines_batch(batch_seeds, wm_mask, wm_affine)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            self.streamline_times.extend([elapsed / len(batch_seeds)] * len(batch_seeds))
            all_streamlines.extend(batch_out)
            print(f"Batch {i//args.batch_size+1}: {len(batch_seeds)} streamlines in {elapsed:.2f}s")

        # Save tractogram
        sft = StatefulTractogram(Streamlines(all_streamlines), seed_tract.header, Space.RASMM)
        save_tractogram(sft, args.output_trk, bbox_valid_check=False)
        print(f"Saved output .trk to {args.output_trk}")

        # Visualizations
        all_gen = np.vstack(all_streamlines)
        all_gt = np.vstack(seeds)
        viz_pred = os.path.join(vis_dir, 'predicted_streamlines_vis.png')
        visualize_3d_streamlines(all_gen, all_gen, args.subject, args.bundle, 'testset', viz_pred)
        viz_comp = os.path.join(vis_dir, 'comparison_streamlines_vis.png')
        visualize_3d_streamlines(all_gen, all_gt, args.subject, args.bundle, 'testset', viz_comp)

        # Timing summary
        times = np.array(self.streamline_times)
        print(f"Total: {times.sum():.2f}s, Avg: {times.mean():.2f}s, Min: {times.min():.2f}s, Max: {times.max():.2f}s, Std: {times.std():.2f}s")
        print("Generation complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate streamlines using diffusion model (optimized)")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--seed_trk", required=True)
    parser.add_argument("--output_trk", required=True)
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--wm_loc", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for parallel generation")
    args = parser.parse_args()

    # Attach batch_size to config
    cfg = TrainingConfig
    cfg.name = "TractoDiff"
    cfg.model.type = "diffusion"
    cfg.model.generator_type = "diffusion"
    cfg.device = "cuda:0"
    cfg.batch_size = args.batch_size

    gen = StreamlineGenerator(cfg)
    gen.generate_all_streamlines(args)

if __name__ == "__main__":
    main()
