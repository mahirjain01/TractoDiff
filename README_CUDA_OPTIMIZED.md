# CUDA-Optimized TractoDiff

This repository contains CUDA-optimized versions of the TractoDiff training and inference pipelines. These optimizations are designed to maximize GPU utilization, reduce training time, and improve inference performance.

## Key Features

- **Multi-GPU Training**: Efficiently distributes training across all available GPUs using PyTorch's DistributedDataParallel (DDP)
- **Heterogeneous GPU Support**: Smart workload balancing across GPUs with different memory capacities (11GB + 8GB)
- **Mixed Precision Training**: Uses PyTorch's Automatic Mixed Precision (AMP) for faster training with reduced memory usage
- **CUDA Stream Optimization**: Overlaps data transfer and computation for better GPU utilization
- **cuDNN Optimizations**: Enables cuDNN benchmarking and other optimizations for faster convolutions
- **GPU Monitoring**: Tracks and visualizes GPU utilization, memory usage, and temperature during training/inference
- **Model Parallelism**: Supports splitting large models across multiple GPUs when needed

## Requirements

- PyTorch 1.9+
- CUDA 11.0+
- cuDNN 8.0+
- All original dependencies from the base repository

## Files

- `train_cuda_optimized.py`: Optimized multi-GPU training pipeline
- `inference_cuda_optimized.py`: Optimized GPU inference pipeline
- `gpu_utils.py`: GPU monitoring and benchmarking utilities
- `model_parallel_utils.py`: Utilities for model parallelism across heterogeneous GPUs
- `run_cuda_optimized.sh`: Script to run the optimized training and inference pipelines

## Usage

### Quick Start

To run the optimized training and inference with default settings:

```bash
# Make script executable
chmod +x run_cuda_optimized.sh

# Run the script (interactive mode)
./run_cuda_optimized.sh

# Or specify the mode directly
./run_cuda_optimized.sh train     # Run training only
./run_cuda_optimized.sh inference # Run inference only
./run_cuda_optimized.sh both      # Run training followed by inference
./run_cuda_optimized.sh benchmark # Run benchmarks only
```

### Training

For direct training:

```bash
python train_cuda_optimized.py
```

The training script automatically:
1. Detects all available GPUs
2. Distributes the workload optimally
3. Applies mixed precision and other optimizations
4. Logs metrics with wandb
5. Saves checkpoints for each epoch

### Inference

For direct inference:

```bash
python inference_cuda_optimized.py
```

The inference script automatically:
1. Uses the GPU with the most available memory
2. Optimizes for faster inference with batching
3. Records and reports inference time and memory usage

### GPU Monitoring

To monitor GPU usage during any process:

```python
from gpu_utils import GPUStats

# Start monitoring
monitor = GPUStats(log_dir='gpu_logs')
monitor.start_monitoring()

# Run your training or inference

# Stop monitoring and plot results
monitor.stop_monitoring()
monitor.plot_stats()
```

## Optimizations Applied

### 1. CUDA and cuDNN Optimizations

- Set `torch.backends.cudnn.benchmark = True` for optimal cuDNN algorithm selection
- Enable TF32 precision on Ampere GPUs for better performance with minimal accuracy impact
- Use `channels_last` memory format for better performance on modern GPUs
- Configure optimal memory splits with `PYTORCH_CUDA_ALLOC_CONF`

### 2. Multi-GPU Training

- Distribute training across available GPUs using DistributedDataParallel (DDP)
- Balance workload based on GPU memory capacity for heterogeneous setups
- Implement proper gradient synchronization and weight updates

### 3. Mixed Precision Training

- Use FP16 precision for most operations while maintaining FP32 precision where needed
- Implement gradient scaling to prevent underflow
- Optimize memory usage to fit larger batch sizes

### 4. CUDA Stream Optimization

- Use separate CUDA streams for data transfer and computation
- Overlap data loading, preprocessing, and model computation
- Properly synchronize streams when needed

### 5. Memory Optimization

- Implement efficient memory management
- Preload and cache data when possible
- Reduce CPU-GPU synchronization points
- Optimize tensor allocation and reuse

## Performance Comparison

| Metric                 | Original | Optimized | Improvement |
|------------------------|----------|-----------|-------------|
| Training time/epoch    | (varies) | (varies)  | (varies)    |
| GPU utilization        | (varies) | (varies)  | (varies)    |
| Inference time/sample  | (varies) | (varies)  | (varies)    |
| Max batch size         | (varies) | (varies)  | (varies)    |

## Model Parallelism

For extremely large models or to better utilize heterogeneous GPUs, you can use the model parallelism utilities:

```python
from model_parallel_utils import auto_parallelize_model

# Automatically choose the best parallelization strategy
model = auto_parallelize_model(model, model_size_mb=5000)  # Approximate model size in MB

# Or explicitly use pipeline parallelism
from model_parallel_utils import PipelineParallelModule
model = PipelineParallelModule(model, num_microbatches=8)
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce batch size
2. Use mixed precision training (enabled by default)
3. If using a large model, try model parallelism with `auto_parallelize_model()`
4. Check for memory leaks using `torch.cuda.memory_summary()`

### Training Instability

If training becomes unstable with mixed precision:

1. Check `train_cuda_optimized.py` for gradient clipping settings
2. Consider adjusting the gradient scaling parameters
3. Identify specific layers that might need FP32 precision

### Poor Performance

If performance is worse than expected:

1. Run `nvidia-smi` to check if other processes are using the GPUs
2. Verify that cuDNN is properly installed and detected
3. Try different batch sizes to find the optimal setting
4. Check for data loading bottlenecks using the profiling tools

## Contributing

We welcome contributions to improve the CUDA optimization. Please feel free to submit issues or pull requests.

## License

This project follows the same licensing as the original TractoDiff repository. 