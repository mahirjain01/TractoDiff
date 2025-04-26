#!/bin/bash

# Setup environment
echo "Setting up environment for CUDA-optimized training and inference..."

# Check for CUDA availability
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: NVIDIA driver or GPU not detected. Please ensure CUDA is properly installed."
    exit 1
fi

# Show GPU information
echo "==== Available GPUs ===="
nvidia-smi --format=csv,noheader --query-gpu=index,name,memory.total,driver_version

# Optimize system settings for deep learning
echo "==== Optimizing system settings for deep learning ===="
# Increase shared memory size
# ulimit -l unlimited > /dev/null 2>&1
# Increase number of open files limit
ulimit -n 64000 > /dev/null 2>&1

# Set CUDA environment variables for performance
export CUDA_AUTO_BOOST=1
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API for PyTorch if available

# Set PyTorch environment variables for performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set number of threads for OpenMP
export OMP_NUM_THREADS=8

# Run CUDA optimization analysis
echo "==== Running GPU optimization analysis ===="
python -c "import torch; import gpu_utils; gpu_utils.print_gpu_info(); gpu_utils.optimize_cudnn()"

# Function to run a benchmark test
run_benchmark() {
    echo "==== Running Matrix Multiplication Benchmark ===="
    python -c "import torch; from gpu_utils import CUDABenchmark; CUDABenchmark.benchmark_matmul(sizes=[(2000, 2000)])"
    
    echo "==== Running Memory Transfer Benchmark ===="
    python -c "import torch; from gpu_utils import CUDABenchmark; CUDABenchmark.benchmark_memory_transfer(sizes=[100])"
}

# Run a quick benchmark
run_benchmark

# Function to select which mode to run
run_mode() {
    echo "Select mode to run:"
    echo "1. Training"
    echo "2. Inference"
    echo "3. Both (Training followed by Inference)"
    echo "4. Benchmark Only"
    read -p "Enter choice [1-4]: " choice
    
    case $choice in
        1)
            run_training
            ;;
        2)
            run_inference
            ;;
        3)
            run_training
            run_inference
            ;;
        4)
            run_benchmark
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
}

# Function to run training
run_training() {
    echo "==== Starting CUDA-Optimized Training ===="
    
    # Start GPU monitoring
    python -c "import gpu_utils; monitor = gpu_utils.GPUStats(log_dir='gpu_logs'); monitor.start_monitoring(); print('GPU monitoring started in background. Will generate reports after training.')"
    
    # Run training
    START_TIME=$(date +%s)
    python train_cuda_optimized.py
    END_TIME=$(date +%s)
    
    # Generate GPU usage plot
    python -c "import gpu_utils; monitor = gpu_utils.GPUStats(log_dir='gpu_logs'); monitor.stop_monitoring(); monitor.plot_stats()"
    
    # Print training time
    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "Training completed in ${ELAPSED_TIME} seconds"
}

# Function to run inference
run_inference() {
    echo "==== Starting CUDA-Optimized Inference ===="
    
    # Start GPU monitoring
    python -c "import gpu_utils; monitor = gpu_utils.GPUStats(log_dir='gpu_logs_inference'); monitor.start_monitoring(); print('GPU monitoring started in background. Will generate reports after inference.')"
    
    # Run inference
    START_TIME=$(date +%s)
    python inference_cuda_optimized.py
    END_TIME=$(date +%s)
    
    # Generate GPU usage plot
    python -c "import gpu_utils; monitor = gpu_utils.GPUStats(log_dir='gpu_logs_inference'); monitor.stop_monitoring(); monitor.plot_stats()"
    
    # Print inference time
    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "Inference completed in ${ELAPSED_TIME} seconds"
}

# Get command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, run interactive mode
    run_mode
else
    # Arguments provided
    case "$1" in
        "train")
            run_training
            ;;
        "inference" | "infer")
            run_inference
            ;;
        "both")
            run_training
            run_inference
            ;;
        "benchmark")
            run_benchmark
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [train|inference|both|benchmark]"
            exit 1
            ;;
    esac
fi

echo "==== Script execution completed ====" 