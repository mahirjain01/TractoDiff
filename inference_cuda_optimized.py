import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.amp as amp
import time
import numpy as np
from tqdm import tqdm
import argparse

# Enable cuDNN benchmarking for optimal performance
torch.backends.cudnn.benchmark = True

def setup_inference_environment():
    """
    Set environment variables and ensure CUDA is available
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU.")
        return torch.device("cpu")
    
    # Get GPU count and memory info
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} CUDA device(s):")
    
    total_memory = 0
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        total_memory += props.total_memory
        print(f"  GPU {i}: {props.name}, {mem_gb:.2f} GB")
    
    print(f"Total GPU memory: {total_memory / (1024**3):.2f} GB")
    
    # Use the GPU with the most memory for inference by default
    max_mem = 0
    best_device = 0
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        if props.total_memory > max_mem:
            max_mem = props.total_memory
            best_device = i
    
    # Set to best device
    device = torch.device(f"cuda:{best_device}")
    torch.cuda.set_device(device)
    print(f"Using GPU {best_device} for inference.")
    
    return device

def load_model(config, device):
    """
    Load the model and move it to the specified device
    """
    from src.models.model import get_model
    
    # Create model
    model = get_model(config=config.model, device="cpu")
    
    # Load checkpoint if specified
    if config.snapshot:
        print(f"Loading weights from {config.snapshot}")
        state_dict = torch.load(config.snapshot, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            model_dict = state_dict['state_dict']
        else:
            model_dict = state_dict
            
        model.load_state_dict(model_dict, strict=False)
        
        # Print model loading info
        print(f"Model loaded from epoch {state_dict.get('epoch', 'unknown')}")
    
    # Move model to device and optimize memory layout
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def preprocess_batch(batch, device):
    """
    Efficiently preprocess and move batch data to device
    """
    # Use non_blocking=True for asynchronous transfers
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    elif isinstance(batch, list):
        return [x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x 
                for x in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    else:
        return batch

def run_inference(model, data_loader, device, output_dir=None, batch_size=None, loss_func=None):
    """
    Run inference with optimized GPU operations
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Create streams for overlapping data transfer and computation
    data_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    
    # Create batches if batch_size is specified and data_loader is not already batched
    if batch_size and not hasattr(data_loader, 'batch_size'):
        # This depends on the data_loader structure, adjust as needed
        pass
    
    # Setup for mixed precision inference
    results = []
    inference_times = []
    
    # Cache CUDA memory for faster allocation
    torch.cuda.empty_cache()
    
    # Track GPU memory usage
    max_memory_allocated = 0
    
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(data_loader, desc="Running inference")):
            batch_start = time.time()
            
            # Prefetch data to GPU using a separate stream
            with torch.cuda.stream(data_stream):
                data_dict = preprocess_batch(data_dict, device)
            
            # Wait for data to be ready
            torch.cuda.current_stream().wait_stream(data_stream)
            
            # Execute model on the compute stream
            with torch.cuda.stream(compute_stream):
                # Mixed precision inference
                with amp.autocast():
                    # Forward pass
                    outputs = model(data_dict)
                
                # Calculate loss if a loss function is provided
                if loss_func:
                    loss = loss_func(outputs, data_dict)
                
                # Record results
                if isinstance(outputs, dict):
                    # Convert outputs back to CPU
                    cpu_outputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                                 for k, v in outputs.items()}
                    results.append(cpu_outputs)
                else:
                    results.append(outputs.detach().cpu())
            
            # Wait for computation to finish
            torch.cuda.current_stream().wait_stream(compute_stream)
            
            # Measure inference time
            batch_end = time.time()
            inference_times.append(batch_end - batch_start)
            
            # Track max memory
            current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
            max_memory_allocated = max(max_memory_allocated, current_memory)
            
            # Optional visualization/saving of results
            if output_dir and hasattr(model, 'visualize_results'):
                model.visualize_results(outputs, data_dict, 
                                        os.path.join(output_dir, f"result_{batch_idx}.png"))
    
    # Print inference statistics
    avg_time = np.mean(inference_times)
    total_time = np.sum(inference_times)
    print(f"Inference complete:")
    print(f"  Average time per batch: {avg_time:.4f}s")
    print(f"  Total inference time: {total_time:.4f}s")
    print(f"  Max GPU memory used: {max_memory_allocated:.2f} MB")
    
    return results

class InferenceOptimized:
    def __init__(self, config):
        """
        Optimized inference class
        Args:
            config: Configuration object containing inference parameters
        """
        self.config = config
        self.device = setup_inference_environment()
        
        # Load model
        self.model = load_model(config, self.device)
        
        # Set up loss function for evaluation metrics if needed
        from src.loss import Loss
        self.loss_func = Loss(cfg=config.loss).to(self.device)
        
        # Load dataset
        from src.data_loader.data_loader import evaluation_data_loader
        self.data_loader = evaluation_data_loader(cfg=config.data)
        
        # Setup output directory
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Inference will output results to: {self.output_dir}")
    
    def run(self):
        """
        Run inference on the evaluation dataset
        """
        print("Starting optimized inference pipeline...")
        
        # Record start time
        start_time = time.time()
        
        # Run inference
        results = run_inference(
            model=self.model,
            data_loader=self.data_loader,
            device=self.device,
            output_dir=self.output_dir,
            loss_func=self.loss_func
        )
        
        # Post-process and save results
        self.process_results(results)
        
        # Report total time
        total_time = time.time() - start_time
        print(f"Total inference pipeline completed in {total_time:.2f} seconds")
        
        return results
    
    def process_results(self, results):
        """
        Process and save inference results
        """
        # This needs to be customized based on the specific model output format
        print(f"Processing {len(results)} result batches...")
        
        # Example: Combine results and save
        # Save the combined results
        import pickle
        results_path = os.path.join(self.output_dir, "inference_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {results_path}")
        
        # Visualize some results for verification
        self.verify_results()
    
    def verify_results(self):
        """
        Verify inference results with visualization
        """
        try:
            # Check if any visualization files were created
            import glob
            vis_files = glob.glob(os.path.join(self.output_dir, "*.png"))
            if vis_files:
                print(f"Found {len(vis_files)} visualization files.")
                print(f"First visualization file: {vis_files[0]}")
                
                # Load and verify first image
                import cv2
                import numpy as np
                img = cv2.imread(vis_files[0])
                
                if img is not None:
                    # Check for expected colors in visualization
                    # This should be customized based on the visualization scheme
                    yellow_mask = np.all(img == [0, 255, 255], axis=-1)  # BGR format
                    cyan_mask = np.all(img == [255, 255, 0], axis=-1)    # BGR format
                    
                    has_yellow = np.any(yellow_mask)
                    has_cyan = np.any(cyan_mask)
                    
                    print("Visualization check:")
                    print(f"  Found ground truth (yellow): {has_yellow}")
                    print(f"  Found prediction (cyan): {has_cyan}")
        except Exception as e:
            print(f"Could not verify visualizations: {str(e)}")
    
    def cleanup(self):
        """
        Clean up resources
        """
        # Free CUDA memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'loss_func'):
            del self.loss_func
            
        torch.cuda.empty_cache()
        print("Inference resources cleaned up.")

def main():
    """
    Main function for inference
    """
    # Import configuration utilities
    from src.utils.configs import TrainingConfig
    from src.utils.arguments import get_configuration
    
    # Get configuration
    config = get_configuration()
    
    # Override configuration for inference
    config.output_dir = os.path.join(os.getcwd(), "inference_results")
    
    # Create inferencer
    inferencer = InferenceOptimized(config)
    
    # Run inference
    try:
        results = inferencer.run()
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        inferencer.cleanup()

if __name__ == "__main__":
    main() 