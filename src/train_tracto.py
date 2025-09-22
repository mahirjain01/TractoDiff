
import os
import torch
import time
import numpy as np

from tqdm import tqdm
import os.path as osp
from src.loss_3d import Loss3D
from datetime import timedelta
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from src.models.model import get_model
# from timm.optim import create_optimizer_v2
from torch.utils.tensorboard import SummaryWriter
from src.data_loader.dataset_tracto import get_dataloaders
from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils.functions import to_device, get_device, release_cuda
from src.utils.configs import ScheduleMethods, LossNames, DataDict

from src.utils.logger import TrainingLogger

class TractographyTrainer:
    def __init__(self, cfgs):
        """
        Trainer class for tractography model
        Args:
            cfgs: Training configuration
        """
        self.name = cfgs.name
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.output_dir = os.path.join(cfgs.output_dir, self.name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.iteration = 0
        self.epoch = 0
        self.training = False

        self.num_val_batches_to_log = 3
        self.gradient_accumulation_steps = getattr(cfgs, 'gradient_accumulation_steps', 4)
        self.use_amp = getattr(cfgs, 'use_amp', True)  
        self.amp_dtype = getattr(cfgs, 'amp_dtype', torch.float16)  
        
        tensorboard_log_dir = os.path.join(self.output_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        self.logger = TrainingLogger(
            output_dir=self.output_dir,
            experiment_name=self.name,
            tensorboard_writer=self.writer
        )

        self.logging = self.logger.event_logger

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
        self.model = get_model(config=cfgs.model, device=self.device, logger = self.logging)
        self.snapshot = cfgs.snapshot
        
        # //////////////////////////////////////////////////////// Weight Initialisation using a pretrained model //////////////////////////////////////////////////////////
        
        # self.logging.info(f"Initializing model with pre-trained weights")
        # # Load the checkpoint file
        # pretrained_checkpoint = torch.load(r"/med/TractoDiff/best_model.pth", map_location=self.device)
        # # Your checkpoints save the model weights under the key 'state_dict'
        # if 'state_dict' in pretrained_checkpoint:
        #     model_weights = pretrained_checkpoint['state_dict']
        # else:
        #     # Handle cases where the .pth file might just be the weights directly
        #     model_weights = pretrained_checkpoint
        # # Load the weights into the model
        # self.model.load_state_dict(model_weights)
        # self.logging.info("Successfully loaded pre-trained weights into the model.")
        
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
        # Setup GPU/distributed training
        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)
            
        self._ensure_model_on_device()

        # Setup logging
        configs = {
            "lr": cfgs.lr,
            "lr_t0": cfgs.lr_t0,
            "lr_tm": cfgs.lr_tm,
            "lr_min": cfgs.lr_min,
            "gpus": cfgs.gpus,
            "epochs": self.max_epoch
        }
        
        self.logging.info(f"The configs being used are : {configs}")
        
         # Initialize gradient scaler for AMP
        self.scaler = GradScaler(device = self.device) if self.use_amp else None
      
        # Setup optimizer and scheduler
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),    
            lr=cfgs.lr, 
            weight_decay=cfgs.weight_decay
        )

        # self.optimizer = create_optimizer_v2(
        #     self.model.parameters(),
        #     opt='adamw',
        #     lr=cfgs.lr,
        #     weight_decay=cfgs.weight_decay,
        #     betas = (0.9, 0.999),
        #     eps = 1e-8
        # )
        
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
            self.logging.warning(f"[SNAPSHOT] Attempting to load snapshot from: {self.snapshot}")
            state_dict = self.load_snapshot(self.snapshot)
            self.logging.warning(f"[SNAPSHOT] Loaded snapshot keys: {list(state_dict.keys())}")
            if not cfgs.only_model:
                self.load_learning_parameters(state_dict)

        # Setup loss function
        self.loss_func = Loss3D(cfg=cfgs.loss, name = self.name, max_visualizations=30)
        self.loss_func = self.loss_func.to(self.device)

        # datasets:
        self.training_data_loader, self.evaluation_data_loader = get_dataloaders(
            cfg=cfgs.data,
            logger=self.logging
        )
        self.norm_mean = self.training_data_loader.dataset.mean.to(self.device)
        self.norm_std = self.training_data_loader.dataset.std.to(self.device)
        self.logging.info(f"Stored normalization mean: {self.norm_mean.tolist()}")
        self.logging.info(f"Stored normalization std: {self.norm_std.tolist()}")
        # self.training_data_loader = train_data_loader(cfg=cfgs.data, logger = self.logging)
        # self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data, logger = self.logging)
        
        self.best_val_loss = float('inf')
        self.save_period = getattr(cfgs, 'save_period', 5) # Get from config, default to 5
        self.logging.info(f"Models will be saved periodically every {self.save_period} epochs.")

        self.time_step_number = cfgs.model.diffusion.traversable_steps
        
        # Additional output_dir
        self.generator_type = cfgs.model.generator_type
        self.time_step_loss_buffer = []
        self.traversability_threshold = cfgs.traversability_threshold

        self.accumulated_loss = 0.0
        self.accumulation_step = 0

    def step(self, data_dict, train=True, log_trajectory_details=False) -> dict:
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
        self.loss_func = self.loss_func.to(self.device)
        
        if train:
            with autocast(device_type = 'cuda', enabled=False):
                output_dict = self.model(data_dict, sample=False)
                # self.logging.info("Output dict keys : ", output_dict["points"].shape)
                # self.logging.info("Shape of prediction : ", output_dict["prediction"].shape)
                torch.cuda.empty_cache()

                loss_dict = self.loss_func(output_dict, self.norm_mean, self.norm_std)
                output_dict.update(loss_dict)
                loss = output_dict[LossNames.loss] / self.gradient_accumulation_steps

            # self.logging.info("The pred is: ", output_dict["prediction"][0])
            # self.logging.info("The gt is: ", output_dict["points"][0])
            output_dict['scaled_loss'] = loss

        else:
            # For evaluation, pass ground truth for logging purposes
            output_dict = self.model(data_dict, sample=True)
            torch.cuda.empty_cache()
            
            gt_normalized = data_dict['points']
            pred_normalized = output_dict['prediction']
            pred_normalized_poses = torch.cumsum(pred_normalized, dim=1) 
            
            mean = self.norm_mean.view(1, 1, 3)
            std = self.norm_std.view(1, 1, 3)
            
            gt_denormalized = (gt_normalized * std) + mean
            pred_denormalized = (pred_normalized_poses * std) + mean
            
            denormalized_dict = {
                'points': data_dict['points'],        # Ground truth in real-world coords
                'prediction': output_dict['prediction'],  # Prediction in real-world coords
                'subject_id': data_dict['subject_id'], # Pass other info through
                'bundle': data_dict['bundle']
            }
            
            eval_dict = self.loss_func.evaluate(denormalized_dict, self.norm_mean, self.norm_std)
            output_dict.update(eval_dict)
            
            if log_trajectory_details:
                 
                self.logging.info("\n=== Epoch {} Trajectory Comparison ===".format(self.epoch))
                self.logging.info(f"{'Point':>8} {'Ground Truth':>40} {'Prediction':>40} {'Difference':>20}")
                self.logging.info("-" * 110)
                
                for i in range(min(3, gt_denormalized.shape[0])):  # Show first 3 trajectories
                    self.logging.info(f"\nTrajectory {i+1}:")
                    for j in range(gt_denormalized.shape[1]):  # For each point in sequence
                        gt_point = gt_denormalized[i, j].cpu().numpy()
                        pred_point = pred_denormalized[i, j].cpu().numpy()
                        diff = np.abs(gt_point - pred_point)
                        
                        self.logging.info(f"Point {j:2d}: "
                            f"[{gt_point[0]:8.3f}, {gt_point[1]:8.3f}, {gt_point[2]:8.3f}] -> "
                            f"[{pred_point[0]:8.3f}, {pred_point[1]:8.3f}, {pred_point[2]:8.3f}] "
                            f"Diff: [{diff[0]:6.3f}, {diff[1]:6.3f}, {diff[2]:6.3f}]")
                    
                    # Calculate and show trajectory statistics
                    mean_error = np.mean(np.abs(gt_denormalized[i].cpu().numpy() - pred_denormalized[i].cpu().numpy()))
                    self.logging.info(f"Mean Error for Trajectory {i+1}: {mean_error:.3f}")
                
                self.logging.info("\n=== Overall Statistics ===")
                total_mean_error = np.mean(np.abs(gt_denormalized.cpu().numpy() - pred_denormalized.cpu().numpy()))
                self.logging.info(f"Total Mean Error: {total_mean_error:.3f}")
                self.logging.info("=" * 110 + "\n")
            
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
            # self.logging.info\
            print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                  % (self.current_rank, world_size))

            # synchronizes all the threads to reach this point before moving on
            dist.barrier()
        else:
            # self.logging.info\
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        # if cfg.local_rank == 0:
        #     self.logging.info(
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
        self.writer.close()
        if self.distributed:
            dist.destroy_process_group()

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
    
    def run_epoch(self):
        """
        run training epochs
        """
        self.optimizer.zero_grad()
        last_time = time.time()

        total_loss, total_path_dis, total_last_pose_dis, num_batches = 0.0, 0.0, 0.0, 0

        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            self.iteration += 1
            data_dict[DataDict.traversable_step] = self.time_step_number
            
            output_dict = self.step(data_dict=data_dict, train = True)
            torch.cuda.empty_cache()
            scaled_loss = output_dict['scaled_loss']
            self.scaler.scale(scaled_loss).backward()

            if (iteration + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Unscale gradients and step the optimizer
                self.scaler.step(self.optimizer)
                
                # Update the scaler
                self.scaler.update()
                
                # Zero the gradients for the next accumulation cycle
                self.optimizer.zero_grad()

            optimize_time = time.time()
            step_duration_sec = optimize_time - last_time
            last_time = time.time()

            output_dict['step_duration_sec'] = step_duration_sec

            total_loss += output_dict[LossNames.loss].item()
            total_path_dis += output_dict.get(LossNames.path_dis, torch.zeros(1)).item()
            total_last_pose_dis += output_dict.get(LossNames.last_dis, torch.zeros(1)).item()
            num_batches += 1

            self.logger.log_iteration(
                iteration=self.iteration,
                metrics=output_dict,
                epoch=self.epoch
            )

            output_dict = release_cuda(output_dict)
            last_time = time.time()

        if num_batches > 0:
            epoch_metrics = {
                'average_loss': total_loss / num_batches,
                'average_path_distance': total_path_dis / num_batches,
                'average_last_pose_distance': total_last_pose_dis / num_batches,
                'learning_rate': self.scheduler.get_last_lr()[-1]
            }
            self.logger.log_epoch(self.epoch, metrics=epoch_metrics, prefix='Train')

        self.scheduler.step()

    def inference_epoch(self):
        self._ensure_model_on_device()
        device = self.device
        
        self.loss_func.reset_vis_counter()
        
        # Log model configuration and sampling parameters
        self.logging.info("\n===== MODEL EVALUATION CONFIGURATION =====")
        self.logging.info(f"Generator type: {self.generator_type}")
        
        # Get diffusion parameters
        if hasattr(self.model.generator, 'sample_times'):
            self.logging.info(f"Sample times: {self.model.generator.sample_times}")
        if hasattr(self.model.generator, 'time_steps'):
            self.logging.info(f"Time steps: {self.model.generator.time_steps}")
        if hasattr(self.model.generator, 'inference_steps'):
            self.logging.info(f"Inference steps: {self.model.generator.inference_steps}")
        
        # Show model's device and mode
        self.logging.info(f"Model device: {next(self.model.parameters()).device}")
        self.logging.info(f"Model mode: {'eval' if not self.model.training else 'train'}")
        self.logging.info("=========================================\n")
        
        epoch_metrics_list = []
        
        for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                    desc="Evaluation Losses Epoch {}".format(self.epoch))):

            # Ensure input data is on correct device
            data_dict = to_device(data_dict, device=device)
            log_details = (iteration < self.num_val_batches_to_log)
            output_dict = self.step(data_dict, train=False, log_trajectory_details=log_details)
            torch.cuda.synchronize()
            
            epoch_metrics_list.append({
                'path_dis': output_dict[LossNames.evaluate_path_dis].item(),
                'last_pose_dis' : output_dict[LossNames.evaluate_last_dis].item(),
                'total_loss': output_dict[LossNames.loss].item() if LossNames.loss in output_dict else None
            })
            
            output_dict = release_cuda(output_dict)

        if epoch_metrics_list:
            avg_path_dis = np.mean([m['path_dis'] for m in epoch_metrics_list])
            avg_last_pos_dis = np.mean([m['last_pose_dis'] for m in epoch_metrics_list])
            valid_losses = [m['total_loss'] for m in epoch_metrics_list if m['total_loss'] is not None]
            avg_total_loss = np.mean(valid_losses) if valid_losses else 0.0
            
            eval_metrics = {
                'average_loss': avg_total_loss,
                'average_path_distance': avg_path_dis,
                'average_last_pose_distance' : avg_last_pos_dis
            }

            self.logger.log_epoch(self.epoch, metrics=eval_metrics, prefix='Validation')
            return eval_metrics.get('average_loss')
            
    def run(self):
        """
        run the training process
        """
        torch.autograd.set_detect_anomaly(True)
        
        try:
            for self.epoch in range(self.epoch, self.max_epoch, 1):

                self.set_train_mode()
                if self.distributed:
                    self.training_data_loader.sampler.set_epoch(self.epoch)
                    if self.evaluation_freq > 0:
                        self.evaluation_data_loader.sampler.set_epoch(self.epoch)
                        
                self.run_epoch()
                
                val_loss = None
                if (self.evaluation_freq > 0) and (self.epoch + 1) % self.evaluation_freq == 0:
                    self.set_eval_mode()
                    val_loss = self.inference_epoch()
                    
                if not self.distributed or self.current_rank == 0:
                    os.makedirs(f'{self.output_dir}/models', exist_ok=True)
                    
                    # 1. Save the best model based on validation loss
                    if val_loss is not None and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_path = f'{self.output_dir}/models/best_model.pth'
                        self.logging.info(
                            f"\nNew best model found! Loss: {self.best_val_loss:.4f} at epoch {self.epoch}. Saving to {save_path}\n"
                        )
                        self.save_snapshot(save_path)

                    # 2. Save periodically
                    if (self.epoch + 1) % self.save_period == 0:
                        save_path = f'{self.output_dir}/models/epoch_{self.epoch}.pth'
                        self.logging.info(
                            f"\nPeriodic save at epoch {self.epoch}. Saving to {save_path}\n"
                        )
                        self.save_snapshot(save_path)
    
        finally:
            self.logging.info("Training finished. Generating final epoch metrics plots...")
            self.logger.plot_epoch_metrics()

            self.cleanup()
