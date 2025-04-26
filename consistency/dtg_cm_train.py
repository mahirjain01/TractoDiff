"""
Train a consistency model from an existing DTG (Diffusion Trajectory Generator) model.
This script adapts the consistency distillation framework for trajectory generation.
"""

import os
import sys
import copy
import time
from os.path import join
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from datetime import timedelta
import wandb
from easydict import EasyDict as edict
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.arguments import get_args, get_configuration
from src.utils.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes, DataDict
from src.loss import Loss
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader
from src.utils.configs import TrainingConfig, GeneratorType, DiffusionModelType, CRNNType


def extend_args(parser):
    """Extend base argument parser with consistency-specific arguments"""
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, 
                       default="./consistency_output",
                       help="Directory to save models and logs")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Steps between logging")
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Steps between saving checkpoints")
    
    # Consistency training parameters
    parser.add_argument("--teacher_model_path", type=str, required=True, 
                       help="Path to the trained DTG model to use as teacher")
    parser.add_argument("--training_mode", type=str, default="consistency_distillation",
                       choices=["consistency_distillation", "consistency_training", "progdist"],
                       help="Training mode for consistency model")
    
    # EMA and scaling parameters
    parser.add_argument("--target_ema_mode", type=str, default="fixed", 
                       choices=["fixed", "adaptive"], help="EMA mode for target model")
    parser.add_argument("--scale_mode", type=str, default="fixed", 
                       choices=["fixed", "progressive", "progdist"], 
                       help="Scale mode for consistency training")
    parser.add_argument("--start_ema", type=float, default=0.95, 
                       help="Initial EMA decay rate")
    parser.add_argument("--start_scales", type=int, default=40, 
                       help="Initial number of diffusion steps")
    parser.add_argument("--end_scales", type=int, default=40, 
                       help="Final number of diffusion steps (for progressive)")
    parser.add_argument("--total_training_steps", type=int, default=10000, 
                       help="Total training steps")
    parser.add_argument("--distill_steps_per_iter", type=int, default=10000, 
                       help="Number of steps per distillation iteration (for progdist)")
    
    return parser


def extend_configuration(cfg, args):
    """Extend base configuration with consistency-specific settings"""
    cfg.consistency = edict()
    cfg.consistency.training_mode = args.training_mode
    cfg.consistency.target_ema_mode = args.target_ema_mode
    cfg.consistency.scale_mode = args.scale_mode
    cfg.consistency.start_ema = args.start_ema
    cfg.consistency.start_scales = args.start_scales
    cfg.consistency.end_scales = args.end_scales
    cfg.consistency.total_training_steps = args.total_training_steps
    cfg.consistency.distill_steps_per_iter = args.distill_steps_per_iter
    cfg.consistency.teacher_model_path = args.teacher_model_path
    
    # Update training settings
    cfg.max_epoch = args.total_training_steps // len(train_data_loader(cfg=cfg.data))
    cfg.evaluation_freq = args.eval_interval if hasattr(args, 'eval_interval') else 5000
    cfg.lr = args.lr if hasattr(args, 'lr') else 1e-4
    
    # Add logging and saving intervals
    cfg.log_interval = args.log_interval
    cfg.save_interval = args.save_interval
    
    return cfg


class DTGConsistencyTrainer:
    def __init__(self, args, cfg):
        """
        Trainer for consistency model distillation from DTG.
        Args:
            args: Command line arguments
            cfg: Training configuration
        """
        self.cfg = cfg
        self.name = cfg.name + "_consistency"
        self.training_mode = cfg.consistency.training_mode
        self.total_training_steps = cfg.consistency.total_training_steps
        
        # Initialize training state
        self.step = 0
        self.epoch = 0
        self.global_step = 0
        
        # Set up device and distributed training
        self._setup_device()
        self._setup_models()
        
        # Set up EMA and scaling functions
        self.ema_scale_fn = self._create_ema_and_scales_fn()
        
        # Initialize optimizer, loss, and data loaders
        self._setup_training_components()
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_device(self):
        """Set up device and distributed training"""
        if self.cfg.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=self.cfg.gpus.device)
            
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        if 'WORLD_SIZE' in os.environ and self.cfg.gpus.device == "cuda":
            self.distributed = self.cfg.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
        else:
            self.distributed = self.cfg.data.distributed = False
            
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                local_rank = int(os.environ['LOCAL_RANK'])
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend='nccl', init_method='env://', 
                                     timeout=timedelta(seconds=5000))
                self.current_rank = dist.get_rank()
            else:
                self.current_rank = 0
        else:
            self.current_rank = 0
    
    def _setup_models(self):
        """Initialize and set up all models"""
        # Initialize models
        self.teacher_model = get_model(config=self.cfg.model, device=self.device)
        self.student_model = get_model(config=self.cfg.model, device=self.device)
        self.target_model = get_model(config=self.cfg.model, device=self.device)
        
        # Load teacher model
        teacher_state_dict = torch.load(self.cfg.consistency.teacher_model_path, 
                                      map_location=self.device)
        self.teacher_model.load_state_dict(teacher_state_dict['state_dict'], strict=False)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        
        # Initialize student with teacher weights
        for dst, src in zip(self.student_model.parameters(), self.teacher_model.parameters()):
            dst.data.copy_(src.data)
            
        # Initialize target model
        for dst, src in zip(self.target_model.parameters(), self.student_model.parameters()):
            dst.data.copy_(src.data)
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        
        # Setup distributed training if needed
        if self.distributed:
            self.student_model.cuda()
            self.teacher_model.cuda()
            self.target_model.cuda()
            
            self.student_model = DDP(
                self.student_model,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                find_unused_parameters=True
            )
        else:
            self.student_model.to(self.device)
            self.teacher_model.to(self.device)
            self.target_model.to(self.device)
    
    def _setup_training_components(self):
        """Set up optimizer, loss function, and data loaders"""
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters() if not self.distributed 
            else self.student_model.module.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        
        # Loss function
        self.loss_func = Loss(cfg=self.cfg.loss)
        if self.device == "cuda":
            self.loss_func = self.loss_func.cuda()
        else:
            self.loss_func = self.loss_func.to(self.device)
            
        # Data loaders
        self.training_data_loader = train_data_loader(cfg=self.cfg.data)
        self.evaluation_data_loader = evaluation_data_loader(cfg=self.cfg.data)
    
    def _setup_logging(self):
        """Initialize wandb logging"""
        wandb.login(key=self.cfg.wandb_api)
        configs = {
            "training_mode": self.training_mode,
            "lr": self.cfg.lr,
            "start_ema": self.cfg.consistency.start_ema,
            "start_scales": self.cfg.consistency.start_scales,
            "end_scales": self.cfg.consistency.end_scales,
            "total_steps": self.cfg.consistency.total_training_steps,
        }
        if self.distributed:
            self.wandb_run = wandb.init(project=self.name, config=configs, group="DDP")
        else:
            self.wandb_run = wandb.init(project=self.name, config=configs)

    def _create_ema_and_scales_fn(self):
        """Create function that controls EMA decay rate and noise scales during training"""
        def ema_and_scales_fn(step):
            if self.cfg.consistency.target_ema_mode == "fixed" and self.cfg.consistency.scale_mode == "fixed":
                target_ema = self.cfg.consistency.start_ema
                scales = self.cfg.consistency.start_scales
            elif self.cfg.consistency.target_ema_mode == "fixed" and self.cfg.consistency.scale_mode == "progressive":
                target_ema = self.cfg.consistency.start_ema
                scales = int(self.cfg.consistency.start_scales + (self.cfg.consistency.end_scales - self.cfg.consistency.start_scales) * 
                            (step / self.cfg.consistency.total_training_steps))
                scales = max(scales, 1)
            elif self.cfg.consistency.target_ema_mode == "adaptive" and self.cfg.consistency.scale_mode == "progressive":
                scales = int(self.cfg.consistency.start_scales + (self.cfg.consistency.end_scales - self.cfg.consistency.start_scales) * 
                            (step / self.cfg.consistency.total_training_steps))
                scales = max(scales, 1)
                # Adjust EMA rate based on number of scales
                c = -torch.log(torch.tensor(self.cfg.consistency.start_ema)) * self.cfg.consistency.start_scales
                target_ema = torch.exp(-c / scales).item()
            elif self.cfg.consistency.target_ema_mode == "fixed" and self.cfg.consistency.scale_mode == "progdist":
                distill_stage = step // self.cfg.consistency.distill_steps_per_iter
                scales = self.cfg.consistency.start_scales // (2**distill_stage)
                scales = max(scales, 2)
                
                if scales == 2:
                    sub_stage = max(step - self.cfg.consistency.distill_steps_per_iter * 
                                   (torch.log2(torch.tensor(self.cfg.consistency.start_scales).float()).item() - 1), 0)
                    sub_stage = sub_stage // (self.cfg.consistency.distill_steps_per_iter * 2)
                    sub_scales = 2 // (2**sub_stage)
                    sub_scales = max(sub_scales, 1)
                    scales = sub_scales if scales == 2 else scales
                
                target_ema = 1.0
            else:
                raise ValueError(f"Unsupported combination of target_ema_mode={self.cfg.consistency.target_ema_mode} "
                                f"and scale_mode={self.cfg.consistency.scale_mode}")
                
            return float(target_ema), int(scales)
            
        return ema_and_scales_fn
    
    def _update_target_model(self):
        """Update target model using EMA of student model parameters"""
        target_ema, _ = self.ema_scale_fn(self.global_step)
        
        # Get source parameters (unwrap from DDP if needed)
        if self.distributed:
            source_params = self.student_model.module.parameters()
        else:
            source_params = self.student_model.parameters()
            
        # Update target model parameters
        with torch.no_grad():
            for target_param, source_param in zip(self.target_model.parameters(), source_params):
                target_param.data.mul_(target_ema).add_(source_param.data, alpha=1 - target_ema)
    
    def _reset_for_progdist(self):
        """Reset training for progressive distillation when scaling changes"""
        if self.training_mode != "progdist":
            return
            
        if self.global_step > 0:
            _, scales_now = self.ema_scale_fn(self.global_step)
            _, scales_prev = self.ema_scale_fn(self.global_step - 1)
            
            if scales_now != scales_prev:
                print(f"Resetting for progressive distillation. Scale changed from {scales_prev} to {scales_now}")
                
                # Copy student model to teacher
                with torch.no_grad():
                    for teacher_param, student_param in zip(
                        self.teacher_model.parameters(),
                        self.student_model.module.parameters() if self.distributed else self.student_model.parameters()
                    ):
                        teacher_param.data.copy_(student_param.data)
                
                # Reset optimizer
                self.optimizer = torch.optim.Adam(
                    self.student_model.parameters() if not self.distributed 
                    else self.student_model.module.parameters(),
                    lr=self.cfg.lr,
                    weight_decay=self.cfg.weight_decay
                )
                
                self.step = 0
    
    def consistency_loss(self, data_dict):
        """
        Compute consistency loss between student and target/teacher predictions
        Args:
            data_dict: Input data dictionary
        Returns:
            Loss dictionary
        """
        # Get current noise scale
        _, num_scales = self.ema_scale_fn(self.global_step)
        
        # Move data to device
        data_dict = to_device(data_dict, self.device)
        data_dict[DataDict.traversable_step] = num_scales  # Set noise scale
        
        if self.training_mode == "progdist":
            # Progressive distillation loss
            if num_scales == self.ema_scale_fn(0)[1]:
                # First stage: student learns from teacher
                with torch.no_grad():
                    # Get teacher prediction
                    teacher_data = copy.deepcopy(data_dict)
                    teacher_output = self.teacher_model(teacher_data, sample=False)
                
                # Get student prediction
                student_output = self.student_model(data_dict, sample=False)
                
                # Compute distillation loss
                student_output.update({
                    "teacher_prediction": teacher_output[DataDict.prediction]
                })
                
                loss_dict = self.loss_func.consistency_loss(
                    student_output, 
                    teacher_model=True,
                    num_scales=num_scales
                )
            else:
                # Later stages: student learns from target model (self-distillation)
                with torch.no_grad():
                    # Get target model prediction
                    target_data = copy.deepcopy(data_dict)
                    target_output = self.target_model(target_data, sample=False)
                
                # Get student prediction
                student_output = self.student_model(data_dict, sample=False)
                
                # Compute self-distillation loss
                student_output.update({
                    "target_prediction": target_output[DataDict.prediction]
                })
                
                loss_dict = self.loss_func.consistency_loss(
                    student_output, 
                    teacher_model=False,
                    num_scales=num_scales
                )
                
        elif "consistency" in self.training_mode:
            # Consistency distillation/training
            with torch.no_grad():
                # Get target model prediction
                target_data = copy.deepcopy(data_dict)
                target_output = self.target_model(target_data, sample=False)
                
                if self.training_mode == "consistency_distillation":
                    # Also get teacher prediction for distillation
                    teacher_data = copy.deepcopy(data_dict)
                    teacher_output = self.teacher_model(teacher_data, sample=False)
            
            # Get student prediction
            student_output = self.student_model(data_dict, sample=False)
            
            # Update with reference predictions
            student_output.update({
                "target_prediction": target_output[DataDict.prediction]
            })
            
            if self.training_mode == "consistency_distillation":
                student_output.update({
                    "teacher_prediction": teacher_output[DataDict.prediction]
                })
                
            # Compute consistency loss
            loss_dict = self.loss_func.consistency_loss(
                student_output,
                teacher_model=(self.training_mode == "consistency_distillation"),
                num_scales=num_scales
            )
        else:
            raise ValueError(f"Unsupported training mode: {self.training_mode}")
            
        return student_output, loss_dict
    
    def train_step(self, data_dict):
        """Run one training step"""
        self.optimizer.zero_grad()
        
        # Compute consistency loss
        output_dict, loss_dict = self.consistency_loss(data_dict)
        
        # Backward pass and optimize
        loss_dict[LossNames.loss].backward()
        self.optimizer.step()
        
        # Update target model
        self._update_target_model()
        
        # Check if we need to reset for progressive distillation
        self._reset_for_progdist()
        
        # Return combined dict
        output_dict.update(loss_dict)
        return output_dict
    
    def eval_step(self, data_dict):
        """Run one evaluation step"""
        with torch.no_grad():
            # Move data to device
            data_dict = to_device(data_dict, self.device)
            
            # Run student model in sampling mode
            output_dict = self.student_model(data_dict, sample=True)
            
            # Compute evaluation metrics
            eval_dict = self.loss_func.evaluate(output_dict)
            output_dict.update(eval_dict)
            
        return output_dict
        
    def train_epoch(self):
        """Run one training epoch"""
        self.student_model.train()
        
        for data_dict in tqdm(self.training_data_loader, desc=f"Training Epoch {self.epoch}"):
            # Train step
            start_time = time.time()
            output_dict = self.train_step(data_dict)
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            
            # Update steps and logging
            self.step += 1
            self.global_step += 1
            
            # Log results
            if self.global_step % self.cfg.log_interval == 0:
                output_dict = release_cuda(output_dict)
                self.update_log(output_dict, step_time, log_name=LogTypes.train)
                
            # Save model
            if self.global_step % self.cfg.save_interval == 0:
                self.save_snapshot()
                
            # Evaluate model
            if self.global_step % self.cfg.evaluation_freq == 0:
                self.evaluate()
                self.student_model.train()
    
    def evaluate(self):
        """Evaluate the model"""
        self.student_model.eval()
        
        for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader, 
                                              desc=f"Evaluating at step {self.global_step}")):
            # Eval step
            start_time = time.time()
            output_dict = self.eval_step(data_dict)
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            
            # Log results
            output_dict = release_cuda(output_dict)
            self.update_log(output_dict, step_time, log_name=LogTypes.others)
            
            # Only evaluate a few batches
            if iteration >= 5:
                break
    
    def update_log(self, results, step_time=None, log_name=None):
        """Update logs with results"""
        if step_time is not None:
            self.wandb_run.log({LogNames.step_time: step_time})
            
        if log_name is None:
            for key, value in results.items():
                self.wandb_run.log({key: value})
        else:
            for key, value in results.items():
                self.wandb_run.log({f"{log_name}/{key}": value})
                
        # Also log current scale and EMA values
        target_ema, num_scales = self.ema_scale_fn(self.global_step)
        self.wandb_run.log({
            "training/target_ema": target_ema,
            "training/num_scales": num_scales,
            "training/global_step": self.global_step
        })
    
    def save_snapshot(self):
        """Save model snapshot"""
        if self.distributed and self.current_rank != 0:
            return
            
        os.makedirs(join(self.cfg.output_dir, 'models'), exist_ok=True)
        
        # Get state dicts
        if self.distributed:
            student_state_dict = self.student_model.module.state_dict()
        else:
            student_state_dict = self.student_model.state_dict()
            
        target_state_dict = self.target_model.state_dict()
        
        # Save student model
        state_dict = {
            'state_dict': student_state_dict,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'training_mode': self.training_mode,
            'ema_rate': self.ema_scale_fn(self.global_step)[0],
            'num_scales': self.ema_scale_fn(self.global_step)[1]
        }
        torch.save(state_dict, join(self.cfg.output_dir, 'models', f'{self.name}_{self.global_step}.pth'))
        
        # Save target model
        target_state_dict = {
            'state_dict': target_state_dict,
            'global_step': self.global_step,
            'ema_rate': self.ema_scale_fn(self.global_step)[0],
            'num_scales': self.ema_scale_fn(self.global_step)[1]
        }
        torch.save(target_state_dict, join(self.cfg.output_dir, 'models', f'{self.name}_target_{self.global_step}.pth'))
        
        # Save snapshot for resuming
        torch.save(state_dict, join(self.cfg.output_dir, f'{self.name}_snapshot.pth.tar'))
    
    def run(self):
        """Run training"""
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        while self.global_step < self.total_training_steps:
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
            
            self.train_epoch()
            self.epoch += 1
            
        # Final evaluation and save
        self.evaluate()
        self.save_snapshot()
        
        # Cleanup
        if self.distributed:
            dist.destroy_process_group()
        self.wandb_run.finish()


def main():
    # Create a completely independent parser for consistency-specific args
    parser = argparse.ArgumentParser(description='Consistency Model Training')
    
    # Add all the arguments we need
    # Base training arguments
    parser.add_argument('--name', type=str, default="dtg", help="name of project")
    parser.add_argument('--wandb_api', type=str, default="db0123ab9f0948cf1cf4cbb182e78069983fc0ba", help="Your wandb api")
    parser.add_argument('--data_root', type=str, help='root of the dataset', default="data_sample")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--workers', type=int, default=16, help="number of workers")
    parser.add_argument('--generator_type', type=int, default=0, help="0: diffusion; 1: cvae")
    parser.add_argument('--diffusion_model', type=int, default=0, help="0: rnn; 1: unet")
    parser.add_argument('--crnn_type', type=int, default=0, help="0: gru; 1: lstm")
    parser.add_argument('--device', type=int, default=-1, help="the gpu id")
    
    # Consistency-specific arguments
    parser = extend_args(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get configuration from scratch, rather than using get_configuration()
    cfg = TrainingConfig
    
    # Set basic config
    cfg.name = args.name
    cfg.wandb_api = args.wandb_api
    cfg.data.name = args.name
    cfg.data.root = args.data_root
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.workers
    cfg.output_dir = args.output_dir
    cfg.lr = args.lr
    
    # Add logging and saving intervals
    cfg.log_interval = args.log_interval
    cfg.save_interval = args.save_interval
    
    # Set model config 
    cfg.model.generator_type = GeneratorType.diffusion if args.generator_type == 0 else GeneratorType.cvae
    cfg.loss.generator_type = cfg.model.generator_type
    
    if args.diffusion_model == 0:
        cfg.model.diffusion.model_type = DiffusionModelType.crnn
        if args.crnn_type == 0:
            cfg.model.diffusion.crnn.type = CRNNType.gru
        else:
            cfg.model.diffusion.crnn.type = CRNNType.lstm
    else:
        cfg.model.diffusion.model_type = DiffusionModelType.unet
    
    # Set device
    if args.device >= 0:
        cfg.gpus.device = f"cuda:{args.device}"
    elif args.device == -1:
        cfg.gpus.device = "cuda"
    else:
        cfg.gpus.device = "cpu"
        
    # Extend with consistency settings
    cfg = extend_configuration(cfg, args)
    
    # Create and run trainer
    trainer = DTGConsistencyTrainer(args, cfg)
    trainer.run()


if __name__ == "__main__":
    main()