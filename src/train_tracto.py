import copy
import os
import time
from os.path import join, exists
from typing import Tuple
from datetime import timedelta

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
import numpy as np

from src.utils.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes, DataDict
from src.loss import Loss
from src.loss_3d import Loss3D
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda

from src.utils.logger import TrainingLogger
# from src.data_loader.dataset_tracto import TractographyDataset, get_dataloader
from src.data_loader.dataset_tracto import train_data_loader, evaluation_data_loader


class TractographyTrainer:
    def __init__(self, cfgs: TrainingConfig):
        """
        Trainer class for tractography model
        Args:
            cfgs: Training configuration
        """
        self.name = cfgs.name
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.train_time_steps = cfgs.train_time_steps

        self.output_dir = "/tracto/TractoDiff/logs"

        self.iteration = 0
        self.epoch = 0
        self.training = False

        self.logger = TrainingLogger(
            output_dir=self.output_dir,
            experiment_name=self.name
        )

        # Set up device
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
            print("The device is: ", self.device)
        else:
            self.device = get_device(device=cfgs.gpus.device)
        
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        # Handle distributed training setup
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            self.distributed = cfgs.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
        else:
            self.distributed = cfgs.data.distributed = False

        # Initialize model
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot
        
        # Setup GPU/distributed training
        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)
            
        self._ensure_model_on_device()

        # Setup logging
        self.output_dir = cfgs.output_dir
        configs = {
            "lr": cfgs.lr,
            "lr_t0": cfgs.lr_t0,
            "lr_tm": cfgs.lr_tm,
            "lr_min": cfgs.lr_min,
            "gpus": cfgs.gpus,
            "epochs": self.max_epoch
        }
        wandb.login(key=cfgs.wandb_api)
        if self.distributed:
            self.wandb_run = wandb.init(project=self.name, config=configs, group="DDP")
        else:
            self.wandb_run = wandb.init(project=self.name, config=configs)

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=cfgs.lr, 
            weight_decay=cfgs.weight_decay
        )
        
        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                cfgs.lr_decay_steps, 
                gamma=cfgs.lr_decay
            )
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                eta_min=cfgs.lr_min,
                T_0=cfgs.lr_t0, 
                T_mult=cfgs.lr_tm
            )
        else:
            raise ValueError("Unsupported scheduler type")

        if self.snapshot:
            print(f"[SNAPSHOT] Attempting to load snapshot from: {self.snapshot}")
            state_dict = self.load_snapshot(self.snapshot)
            print(f"[SNAPSHOT] Loaded snapshot keys: {list(state_dict.keys())}")
            if not cfgs.only_model:
                self.load_learning_parameters(state_dict)

        # Setup loss function
        self.loss_func = Loss3D(cfg=cfgs.loss)
        self.loss_func = self.loss_func.to(self.device)

        # datasets:
        self.training_data_loader = train_data_loader(cfg=cfgs.data)
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data)

        # Additional output_dir
        self.use_traversability = cfgs.loss.use_traversability
        self.generator_type = cfgs.model.generator_type
        self.time_step_loss_buffer = []
        self.time_step_number = cfgs.model.diffusion.traversable_steps
        self.traversability_threshold = cfgs.traversability_threshold

    # Keep all the existing methods from train.py, but modify step() to handle the new dataset format
    def step(self, data_dict, train=True) -> dict:
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
        
        if train:
            output_dict = self.model(data_dict, sample=False)
            # print("Output dict keys : ", output_dict["points"].shape)
            # print("Shape of prediction : ", output_dict["prediction"].shape)
            torch.cuda.empty_cache()
            self.loss_func = self.loss_func.to(self.device)

            y_hat = output_dict["prediction"][0]
            y_hat_poses = y_hat
           
            print("The pred is: ", y_hat_poses)
            print("The gt is: ", output_dict["points"][0])
            
            loss_dict = self.loss_func(output_dict)
            output_dict.update(loss_dict)
        else:
            # For evaluation, pass ground truth for logging purposes
            output_dict = self.model(data_dict, sample=True)
            torch.cuda.empty_cache()
            self.loss_func = self.loss_func.to(self.device)
            eval_dict = self.loss_func.evaluate(output_dict)
            
            # New comparison logging code
            gt = data_dict['points']
            pred = output_dict['prediction']
            
            print("\n=== Epoch {} Trajectory Comparison ===".format(self.epoch))
            print(f"{'Point':>8} {'Ground Truth':>40} {'Prediction':>40} {'Difference':>20}")
            print("-" * 110)
            
            for i in range(min(3, gt.shape[0])):  # Show first 3 trajectories
                print(f"\nTrajectory {i+1}:")
                for j in range(gt.shape[1]):  # For each point in sequence
                    gt_point = gt[i, j].cpu().numpy()
                    pred_point = pred[i, j].cpu().numpy()
                    diff = np.abs(gt_point - pred_point)
                    
                    print(f"Point {j:2d}: "
                          f"[{gt_point[0]:8.3f}, {gt_point[1]:8.3f}, {gt_point[2]:8.3f}] -> "
                          f"[{pred_point[0]:8.3f}, {pred_point[1]:8.3f}, {pred_point[2]:8.3f}] "
                          f"Diff: [{diff[0]:6.3f}, {diff[1]:6.3f}, {diff[2]:6.3f}]")
                
                # Calculate and show trajectory statistics
                mean_error = np.mean(np.abs(gt[i].cpu().numpy() - pred[i].cpu().numpy()))
                print(f"Mean Error for Trajectory {i+1}: {mean_error:.3f}")
            
            print("\n=== Overall Statistics ===")
            total_mean_error = np.mean(np.abs(gt.cpu().numpy() - pred.cpu().numpy()))
            print(f"Total Mean Error: {total_mean_error:.3f}")
            print("=" * 110 + "\n")
            
            output_dict.update(eval_dict)

        # For inference, log diffusion model parameters
        if not self.training and hasattr(self.model, 'generator') and hasattr(self.model.generator, 'sample'):
            if hasattr(self.model.generator, 'time_steps'):
                print(f"[DEBUG] Diffusion time_steps: {self.model.generator.time_steps}")
            if hasattr(self.model.generator, 'sample_times'):
                print(f"[DEBUG] Diffusion sample_times: {self.model.generator.sample_times}")
    
            
        return output_dict

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
        print(f'[SNAPSHOT] Loading from "{snapshot}".')
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        print(f"[SNAPSHOT] Model state dict keys: {list(model_dict.keys())[:5]} ... (total {len(model_dict)})")
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            print(f'[SNAPSHOT] Warning: Missing keys: {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'[SNAPSHOT] Warning: Unexpected keys: {unexpected_keys}')
        print('[SNAPSHOT] Model has been loaded.')
        return state_dict

    def load_learning_parameters(self, state_dict):
        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch'] 
            print(f'[SNAPSHOT] Epoch has been loaded: {self.epoch}.')
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print(f'[SNAPSHOT] Iteration has been loaded: {self.iteration}.')
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                print('[SNAPSHOT] Optimizer state has been loaded.')
            except Exception as e:
                print(f"[SNAPSHOT] Couldn't load optimizer: {e}")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                print('[SNAPSHOT] Scheduler state has been loaded.')
            except Exception as e:
                print(f"[SNAPSHOT] Couldn't load scheduler: {e}")

    def save_snapshot(self, filename):
        """
        save the snapshot of the model and other training parameters
        Args:
            filename: the output filename that is the full directory
        """
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)
        # print('Model saved to "{}"'.format(filename))

        # save snapshot
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(self.output_dir, str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        # print('Snapshot saved to "{}"'.format(snapshot_filename))

    def cleanup(self):
        if self.distributed:
            dist.destroy_process_group()
        self.wandb_run.finish()

    def set_train_mode(self):
        """
        set the model to the training mode: parameters are differentiable 
        """
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """
        set the model to the evaluation mode: parameters are not differentiable
        """
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def optimizer_step(self):
        """
        run one step of the optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _ensure_model_on_device(self):
        """Helper method to ensure model is on the correct device"""
        if self.distributed:
            return  # Don't change device for distributed training

        if hasattr(self.model, 'device'):
            if str(self.model.device) != str(self.device):
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
    
    def update_log(self, results, timestep=None, log_name=None):
        if timestep is not None:
            self.wandb_run.log({LogNames.step_time: timestep})
        if log_name == LogTypes.train:
            value = self.scheduler.get_last_lr()
            self.wandb_run.log({log_name + "/" + LogNames.lr: value[-1]})

        if log_name is None:
            for key, value in results.items():
                self.wandb_run.log({key: value})
        else:
            for key, value in results.items():
                self.wandb_run.log({log_name + "/" + key: value})

    def run_epoch(self):
        """
        run training epochs
        """
        self.optimizer.zero_grad()

        last_time = time.time()

        total_loss = 0
        num_batches = 0

        # with open(self.output_file, "a") as f:
        #     print("Training CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            self.iteration += 1
            data_dict[DataDict.traversable_step] = self.time_step_number
            for step_iteration in range(self.train_time_steps):
                output_dict = self.step(data_dict=data_dict)
                torch.cuda.empty_cache()

                output_dict[LossNames.loss].backward()
                self.optimizer_step()
                optimize_time = time.time()

                loss = output_dict[LossNames.loss]
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)

                self.logger.log_iteration(
                    iteration=self.iteration,
                    loss=loss_value,
                    epoch=self.epoch
                )

                output_dict = release_cuda(output_dict)
                self.update_log(results=output_dict, timestep=optimize_time - last_time, log_name=LogTypes.train)
                last_time = time.time()

                total_loss += loss_value
                num_batches += 1
        
        epoch_avg_loss = self.logger.log_epoch(self.epoch)

        self.scheduler.step()

        # Plot loss curve after each epoch
        loss_plot_path = os.path.join(self.output_dir, f'loss_curve_epoch_{self.epoch}.png')
        self.logger.plot_losses(save_path=loss_plot_path)
        
        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth'.format(self.output_dir, self.name, self.epoch))

    def inference_epoch(self):
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0):
            # Ensure model and loss function are on correct device
            self._ensure_model_on_device()
            device = self.device
            
            # Log model configuration and sampling parameters
            print("\n===== MODEL EVALUATION CONFIGURATION =====")
            print(f"Generator type: {self.generator_type}")
            
            # Get diffusion parameters
            if hasattr(self.model.generator, 'sample_times'):
                print(f"Sample times: {self.model.generator.sample_times}")
            if hasattr(self.model.generator, 'time_steps'):
                print(f"Time steps: {self.model.generator.time_steps}")
            if hasattr(self.model.generator, 'inference_steps'):
                print(f"Inference steps: {self.model.generator.inference_steps}")
            
            # Show model's device and mode
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model mode: {'eval' if not self.model.training else 'train'}")
            print("=========================================\n")
            
            # Move loss function to correct device
            self.loss_func = self.loss_func.to(device)
            
            epoch_losses = []  # Store losses for this epoch
            
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                       desc="Evaluation Losses Epoch {}".format(self.epoch))):
                start_time = time.time()

                # Ensure input data is on correct device
                data_dict = to_device(data_dict, device=device)
                output_dict = self.step(data_dict, train=False)
                torch.cuda.synchronize()
                step_time = time.time()
                
                # Store loss values
                epoch_losses.append({
                    'path_dis': output_dict[LossNames.evaluate_path_dis].item(),
                    'total_loss': output_dict[LossNames.loss].item() if LossNames.loss in output_dict else None
                })
                
                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                self.update_log(results=output_dict, timestep=step_time - start_time, log_name=LogTypes.others)

    
            # Print summary statistics
            avg_path_dis = np.mean([l['path_dis'] for l in epoch_losses])
            print(f"\nEpoch {self.epoch} Average Path Distance: {avg_path_dis:.4f}")
            if epoch_losses[0]['total_loss'] is not None:
                avg_total_loss = np.mean([l['total_loss'] for l in epoch_losses])
                print(f"Epoch {self.epoch} Average Total Loss: {avg_total_loss:.4f}")

    def run(self):
        """
        run the training process
        """
        torch.autograd.set_detect_anomaly(True)
        try:
            for self.epoch in range(self.epoch, self.max_epoch, 1):

                if(self.epoch != 0):
                    self.set_eval_mode()
                    self.inference_epoch()

                self.set_train_mode()
                if self.distributed:
                    self.training_data_loader.sampler.set_epoch(self.epoch)
                    if self.evaluation_freq > 0:
                        self.evaluation_data_loader.sampler.set_epoch(self.epoch)
                self.run_epoch()
    
        finally:
            # Create final loss plot with a distinctive name
            final_plot_path = os.path.join(self.output_dir, f'{self.name}_final_loss_curve.png')
            self.logger.plot_losses(save_path=final_plot_path)
            self.cleanup()

