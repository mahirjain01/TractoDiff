import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess

from warnings import warn
import torch
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import datetime, timedelta

from src.utils.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes, DataDict, GeneratorType
from src.loss import Loss
from src.loss_3d import Loss3D
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda
from src.data_loader.data_loader import evaluation_data_loader


class Inference:
    def __init__(self, cfgs: TrainingConfig):

        self.evaluation_freq = cfgs.evaluation_freq

        self.name = cfgs.name
        self.iteration = 0
        self.epoch = 0
        self.training = False
        self.output_dir = cfgs.output_dir

        # set up gpus
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)
        
        # Make sure device is a torch.device object for consistency
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            print("world size: ", int(os.environ['WORLD_SIZE']))
            self.distributed = cfgs.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
        else:
            print("world size: ", 0)
            self.distributed = cfgs.data.distributed = False

        # model
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot
        if self.snapshot:
            state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)
            
        # Verify model is on correct device after setup
        self._ensure_model_on_device()

        # set up loggers
        configs = {
            "lr": cfgs.lr,
            "lr_t0": cfgs.lr_t0,
            "lr_tm": cfgs.lr_tm,
            "lr_min": cfgs.lr_min,
            "gpus": cfgs.gpus,
        }

        self.load_learning_parameters(state_dict)

        # loss functions
        if self.device == "cuda":
            self.loss_func = Loss3D(cfg=cfgs.loss).cuda()
        else:
            self.loss_func = Loss3D(cfg=cfgs.loss).to(self.device)

        # datasets:
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data)
    
        self.use_traversability = cfgs.loss.use_traversability
        self.generator_type = cfgs.model.generator_type
        self.time_step_loss_buffer = []
        self.time_step_number = cfgs.model.diffusion.traversable_steps
        self.traversability_threshold = cfgs.traversability_threshold

    def _set_model_gpus(self, cfg):
        # self.current_rank = 0  # global rank
        # cfg.local_rank = os.environ['LOCAL_RANK']
        if self.distributed:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print("os world size: {}, local_rank: {}, rank: {}".format(world_size, local_rank, rank))

            # this will make all .cuda() calls work properly
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5000))
            # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            # self.logger.info\
            print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                  % (self.current_rank, world_size))

            # synchronizes all the threads to reach this point before moving on
            dist.barrier()
        else:
            # self.logger.info\
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        # if cfg.local_rank == 0:
        #     self.logger.info(
        #         f'Model created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        # move model to GPU, enable channels last layout if set
        if self.distributed:
            self.model.cuda()
        else:
            self.model.to(self.device)

        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        # setup distributed training
        if self.distributed:
            if cfg.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
            self.model = DDP(self.model, device_ids=[cfg.local_rank],
                             broadcast_buffers=not cfg.no_ddp_bb,
                             find_unused_parameters=True)
            # NOTE: EMA model does not need to be wrapped by DDP

        # # setup exponential moving average of model weights, SWA could be used here too
        # model_ema = None
        # if args.model_ema:
        #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        #     model_ema = ModelEmaV2(
        #         self.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

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

    def cleanup(self):
        if self.distributed:
            dist.destroy_process_group()

    def set_eval_mode(self):
        """
        set the model to the evaluation mode: parameters are not differentiable
        """
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def _ensure_model_on_device(self):
        """Helper method to ensure model is on the correct device"""
        if self.distributed:
            return  # Don't change device for distributed training

        if hasattr(self.model, 'device'):
            if str(self.model.device) != str(self.device):
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
        
    def step(self, data_dict) -> dict:
        """
        One step of training/evaluation
        Args:
            data_dict: Dictionary containing:
                - points: (B, 16, 3) tensor of point sequences
                - condition: (B, 334) tensor of condition vectors
            train: Whether this is a training step
        Returns:
            Output dictionary containing model outputs and losses
        """
        self._ensure_model_on_device()
        data_dict = to_device(data_dict, device=self.device)
        
        
        # For evaluation, pass ground truth for logging purposes
        output_dict = self.model(data_dict, sample=True)
        torch.cuda.empty_cache()
        self.loss_func = self.loss_func.to(self.device)
        eval_dict = self.loss_func.evaluate(output_dict)
        
        output_dict.update(eval_dict)

        return output_dict

    def Inference_res(self):
        # Initialize metric accumulators
        total_metrics = {
            'path_distance': 0.0,
            'last_distance': 0.0,
            'traversability': 0.0 if self.use_traversability else None,
            'inference_time': 0.0 
        }
        sample_count = 0
        
        self._ensure_model_on_device()
        device = self.device
        
        self.loss_func = self.loss_func.to(device)
            
        for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                    desc="Running inference...")):
            # Start timing
            start_time = time.time()
            
            # Ensure input data is on correct device
            data_dict = to_device(data_dict, device=device)
            output_dict = self.step(data_dict)
            
            # End timing and synchronize if using GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            
            # Print individual sample inference time
            print(f"\nSample {iteration + 1} inference time: {inference_time:.4f} seconds")
            
            # Accumulate metrics
            total_metrics['path_distance'] += output_dict[LossNames.evaluate_path_dis].item()
            total_metrics['last_distance'] += output_dict[LossNames.evaluate_last_dis].item()
            total_metrics['inference_time'] += inference_time
            if self.use_traversability:
                total_metrics['traversability'] += output_dict[LossNames.evaluate_traversability].item()
            
            sample_count += 1
            
            if iteration == 0:  # Check first batch
                print("Data shapes:")
                print(f"Ground truth: {output_dict[DataDict.points].shape}")
                print(f"Prediction: {output_dict[DataDict.prediction].shape}")
            
            output_dict = release_cuda(output_dict)
            torch.cuda.empty_cache()

        # Calculate averages
        avg_metrics = {
            key: value / sample_count if value is not None else None 
            for key, value in total_metrics.items()
        }
        
        # Save metrics to text file
        metrics_file = os.path.join(self.output_dir, 'average_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Evaluation Results (averaged over {sample_count} samples):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average Path Distance: {avg_metrics['path_distance']:.4f}\n")
            f.write(f"Average Last Position Distance: {avg_metrics['last_distance']:.4f}\n")
            if self.use_traversability:
                f.write(f"Average Traversability Score: {avg_metrics['traversability']:.4f}\n")
            f.write(f"Average Inference Time: {avg_metrics['inference_time']:.4f} seconds\n")
            f.write(f"Total Inference Time: {total_metrics['inference_time']:.4f} seconds\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model snapshot from epoch {self.epoch}, iteration {self.iteration}\n")
            f.write(f"Device used: {self.device}\n")
        
        print(f"\nAverage metrics have been saved to: {metrics_file}")
        print(f"Average inference time per sample: {avg_metrics['inference_time']:.4f} seconds")
        print(f"Total inference time: {total_metrics['inference_time']:.4f} seconds")
        return avg_metrics

    def run(self):
        """
        Run inference on the evaluation dataset
        """
        torch.autograd.set_detect_anomaly(False) 
        self.set_eval_mode()
        self.Inference_res()
        self.cleanup()
