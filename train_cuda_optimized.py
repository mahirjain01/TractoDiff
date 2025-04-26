import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp
import time
import numpy as np
import wandb

# Import required constants
from src.utils.configs import DataDict, LossNames, TrainingConfig
from src.utils.arguments import get_configuration
from src.models.model import get_model


time_step_number = 0
# Enable cuDNN benchmarking for optimal performance
torch.backends.cudnn.benchmark = True
# Enable cuDNN deterministic mode for reproducibility
# torch.backends.cudnn.deterministic = True

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group with NCCL backend (optimized for CUDA)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

 
def get_data_loader(rank, world_size, cfg):
    """
    Create properly sharded data loaders for distributed training
    """
    from src.data_loader.data_loader import train_data_loader
    
    # Set distributed flag based on world_size
    cfg.distributed = world_size > 1
    if cfg.distributed:
        cfg.local_rank = rank
    
    # Get the data loader with proper sharding for distributed training
    loader = train_data_loader(cfg=cfg)
    
    return loader

def validate(model, cfg, val_loader, criterion, device, rank=0):
    """Run validation matching the original inference_epoch method"""
    model.eval()
    
    # Similar to original inference_epoch
    for batch_idx, data_dict in enumerate(val_loader):
        # Move data to device with non-blocking transfer
        data_dict = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in data_dict.items()}
        
        # Ensure the traversable_step is set
        if DataDict.traversable_step not in data_dict:
            data_dict[DataDict.traversable_step] = cfg.model.diffusion.traversable_steps
        
        # Run inference with sample=True as in the original code
        with torch.no_grad():
            # Forward pass - use sample=True for inference as in the original
            output_dict = model(data_dict, sample=True)
            
            # Apply evaluation metrics - use the evaluate method instead of forward
            eval_dict = criterion.evaluate(output_dict)
            output_dict.update(eval_dict)
        
        # No need to calculate a single validation loss
        # Just log the metrics if this is the main process
        if rank == 0 and batch_idx == 0:  # Log only first batch for brevity
            # Log metric values from output_dict as needed
            for key, value in output_dict.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    print(f"  {key}: {value.item():.4f}")
    
    # Return success rather than a specific loss value
    return 0.0  # No single validation loss in original code

def log_gpu_memory(rank):
    """Log GPU memory usage"""
    mem_allocated = torch.cuda.memory_allocated(rank) / (1024 ** 2)  # MB
    mem_reserved = torch.cuda.memory_reserved(rank) / (1024 ** 2)    # MB
    return f"GPU:{rank} Mem: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved"

def step(model, cfg, data_dict, criterion, training=True):
    """
    Perform a single step similar to the original code
    
    Args:
        model: The model
        data_dict: Input data dictionary
        criterion: Loss function
        training: Whether this is a training step or evaluation
    
    Returns:
        Output dictionary with model outputs and loss values
    """
    # Add the traversable_step if needed
    if DataDict.traversable_step not in data_dict:
        if hasattr(model, 'module') and hasattr(model.module, 'diffusion'):
            data_dict[DataDict.traversable_step] = cfg.model.diffusion.traversable_steps
        elif hasattr(model, 'diffusion'):
            data_dict[DataDict.traversable_step] = cfg.model.diffusion.traversable_steps
    
    # Forward pass with appropriate sample parameter based on training mode
    if training:
        output_dict = model(data_dict, sample=False)
        # Apply loss function and update output dictionary
        loss_dict = criterion(output_dict)
        output_dict.update(loss_dict)
        loss_tensor = output_dict[LossNames.loss]
    else:
        output_dict = model(data_dict, sample=True)
        # Use evaluate method if available
        if hasattr(criterion, 'evaluate'):
            eval_dict = criterion.evaluate(output_dict)
            output_dict.update(eval_dict)
    
    return output_dict

def train(rank, world_size, cfg):
    """Main training function optimized for multi-GPU training"""
    # Initialize distributed process group
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Create model and move to GPU
    model = get_model(config=cfg.model, device=device)
    model = model.to(device)
    
    # Use channels_last memory format for better performance on CUDA
    model = model.to(memory_format=torch.channels_last)
    
    # Wrap model with DDP for distributed training
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Store the traversable_step value for consistency
    traversable_steps = cfg.model.diffusion.traversable_steps
    
    # Load optimizer state and checkpoint if specified
    if hasattr(cfg, 'snapshot') and cfg.snapshot:
        checkpoint = torch.load(cfg.snapshot, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if rank == 0:
            print(f"Loaded checkpoint from {cfg.snapshot}")

    # Define loss function
    from src.loss import Loss
    criterion = Loss(cfg=cfg.loss).to(device)
    
    # Define optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr,
        weight_decay=getattr(cfg, 'weight_decay', 0)
    )
    
    # Load optimizer state if available
    if hasattr(cfg, 'snapshot') and cfg.snapshot and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Define learning rate scheduler
    if hasattr(cfg, 'scheduler') and cfg.scheduler:
        if cfg.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=cfg.lr_t0, 
                T_mult=cfg.lr_tm,
                eta_min=cfg.lr_min
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=cfg.lr_decay_steps, 
                gamma=cfg.lr_decay
            )
    else:
        scheduler = None
    
    # Load scheduler state if available
    if hasattr(cfg, 'snapshot') and cfg.snapshot and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Get data loader with proper sharding
    train_loader = get_data_loader(rank, world_size, cfg.data)
    
    # Initialize validation data loader if needed
    if hasattr(cfg, 'evaluation_freq') and cfg.evaluation_freq > 0:
        from src.data_loader.data_loader import evaluation_data_loader
        val_loader = evaluation_data_loader(cfg=cfg.data)
    else:
        val_loader = None
    
    # Initialize mixed precision training
    scaler = amp.GradScaler()
    
    # Pre-allocate CUDA streams for overlapping operations
    # One for data transfers, one for computation
    data_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    
    # Initialize wandb for logging if this is the main process
    if rank == 0:
        wandb.login(key=cfg.wandb_api if hasattr(cfg, 'wandb_api') else None)
        wandb_run = wandb.init(
            project=cfg.name,
            config={
                "lr": cfg.lr,
                "lr_t0": cfg.lr_t0 if hasattr(cfg, 'lr_t0') else None,
                "lr_tm": cfg.lr_tm if hasattr(cfg, 'lr_tm') else None,
                "lr_min": cfg.lr_min if hasattr(cfg, 'lr_min') else None,
                "epochs": cfg.max_epoch,
                "world_size": world_size,
                "optimizer": "Adam",
                "weight_decay": getattr(cfg, 'weight_decay', 0),
                "mixed_precision": True
            },
            group="DDP" if world_size > 1 else None
        )
    
    # Load starting epoch if available
    start_epoch = 0
    if hasattr(cfg, 'snapshot') and cfg.snapshot and 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
    
    # Main training loop
    model.train()
    total_start_time = time.time()
    
    for epoch in range(start_epoch, cfg.max_epoch):
        epoch_start = time.time()
        running_loss = 0.0
        
        # Reset distributed sampler for each epoch
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Use tqdm for progress bar if this is the main process
        if rank == 0:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epoch}")
        else:
            train_iter = train_loader
        
        for i, data_dict in enumerate(train_iter):
            # Prefetch next batch using a separate CUDA stream
            with torch.cuda.stream(data_stream):
                # Move data to device (with non-blocking transfer for performance)
                data_dict = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                             for k, v in data_dict.items()}
                
                # Add the traversable_step key to match the original implementation
                data_dict[DataDict.traversable_step] = traversable_steps
            
            # Wait for the data to be ready
            torch.cuda.current_stream().wait_stream(data_stream)
            
            # Run multiple optimization steps per batch like in the original
            for step_iteration in range(cfg.train_time_steps):
                with torch.cuda.stream(compute_stream):
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    with amp.autocast():
                        output_dict = model(data_dict, sample=False)
                        loss_dict = criterion(output_dict)
                        output_dict.update(loss_dict)
                        loss = output_dict[LossNames.loss]
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Apply gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update running loss
                    running_loss += loss.item()
            
            # Wait for compute to finish before next iteration
            torch.cuda.current_stream().wait_stream(compute_stream)
            
            # Log GPU memory usage occasionally
            if rank == 0 and i % 100 == 0:
                # Log batch time and loss
                if hasattr(train_iter, 'set_postfix'):
                    train_iter.set_postfix(loss=loss.item(), 
                                           lr=optimizer.param_groups[0]['lr'],
                                           mem=log_gpu_memory(rank))
                
                # Log to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                })
        
        # Log epoch metrics
        if rank == 0:
            epoch_time = time.time() - epoch_start
            epoch_loss = running_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}/{cfg.max_epoch}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Log to wandb
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "epoch_time": epoch_time,
            }
            
            # Run validation if needed
            if val_loader is not None and (epoch + 1) % cfg.evaluation_freq == 0:
                print(f"Running validation for epoch {epoch+1}...")
                validate(model, cfg, val_loader, criterion, device, rank)
                
                # No need to update scheduler with validation loss
                # Only update if using ReduceLROnPlateau and you have a specific metric
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use a specific metric if available, or just continue with epoch-based scheduling
                    if 'val_metric' in locals():
                        scheduler.step(val_metric)
            
            wandb.log(log_dict)
            
            # Save checkpoint
            if (epoch + 1) % cfg.save_freq == 0 or epoch == cfg.max_epoch - 1:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if scheduler is not None:
                    checkpoint['scheduler'] = scheduler.state_dict()
                
                checkpoint_dir = os.path.join(cfg.output_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Synchronize processes at the end of each epoch
        dist.barrier()
    
    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"Training completed in {total_time:.2f} seconds")
        wandb.log({"total_training_time": total_time})
        wandb.finish()
    
    # Cleanup
    cleanup()

def get_gpu_info():
    """Get information about available GPUs"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / (1024**2)  # Convert to MiB
        info.append(f"GPU {i}: {props.name}, {mem_total:.0f} MiB")
    
    return "\n".join(info)

def main():
    """Main function to set up distributed training"""
    print("CUDA available:", torch.cuda.is_available())
    print(get_gpu_info())
    
    # Get configuration
    cfg = get_configuration()
    
    # Update config to use CUDA
    if torch.cuda.is_available():
        cfg.gpus.device = "cuda"
    
    # Check for output directory
    if not hasattr(cfg, 'output_dir') or not cfg.output_dir:
        cfg.output_dir = 'outputs'
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Set world_size to the number of GPUs available
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    if world_size < 1:
        print("No GPUs available! Exiting...")
        return
    
    # Set save frequency if not defined
    if not hasattr(cfg, 'save_freq'):
        cfg.save_freq = 1
    
    # Use all available GPUs with DDP
    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' for CUDA support
    mp.set_start_method('spawn', force=True)
    main() 