#!/usr/bin/env python3
"""
CUDA Performance Testing Script

This script benchmarks the CUDA optimizations for training and inference,
comparing against the original implementation. It provides visual reports
and detailed metrics.
"""

import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from gpu_utils import GPUStats, CUDABenchmark, optimize_cudnn, print_gpu_info

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CUDA Performance Testing Script")
    parser.add_argument('--mode', choices=['train', 'inference', 'both', 'benchmark'], 
                       default='benchmark', help='Testing mode')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations for testing')
    parser.add_argument('--output_dir', type=str, default='performance_reports',
                       help='Directory to save performance reports')
    parser.add_argument('--skip_original', action='store_true',
                       help='Skip testing the original implementation')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--model_test', action='store_true',
                       help='Test model parallelism')
    
    return parser.parse_args()

def create_test_model(size='small'):
    """Create a test model of specified size"""
    if size == 'small':
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 8 * 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )
    elif size == 'medium':
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )
    else:  # large
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 2 * 2, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 10)
        )

def create_test_data(batch_size=32, input_size=(3, 32, 32)):
    """Create test data of specified size"""
    return torch.randn(batch_size, *input_size)

def test_training_performance(model, data, target, iterations=5, use_amp=False):
    """Test training performance"""
    if torch.cuda.is_available():
        model = model.to('cuda')
        data = data.to('cuda')
        target = target.to('cuda')
    
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    return avg_time

def test_inference_performance(model, data, iterations=5, use_amp=False):
    """Test inference performance"""
    if torch.cuda.is_available():
        model = model.to('cuda')
        data = data.to('cuda')
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(data)
            else:
                _ = model(data)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(data)
            else:
                _ = model(data)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    return avg_time

def test_model_parallelism(args):
    """Test model parallelism performance"""
    from model_parallel_utils import (
        auto_parallelize_model, 
        ModelParallelModule,
        PipelineParallelModule,
        get_balanced_data_parallel
    )
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping model parallelism tests.")
        return
    
    print("\n=== Testing Model Parallelism ===\n")
    
    # Create a large model
    model = create_test_model('large')
    batch_size = 64
    data = create_test_data(batch_size)
    target = torch.randint(0, 10, (batch_size,))
    
    performance_data = {}
    
    # Test without parallelism (baseline)
    print("Testing baseline (no parallelism)...")
    model_baseline = model.to('cuda:0')
    baseline_time = test_training_performance(model_baseline, data, target, args.iterations)
    performance_data['baseline'] = baseline_time
    print(f"Baseline training time: {baseline_time:.6f} seconds per iteration")
    
    # Test with DataParallel
    print("\nTesting torch.nn.DataParallel...")
    model_dp = torch.nn.DataParallel(model.to('cuda:0'))
    dp_time = test_training_performance(model_dp, data, target, args.iterations)
    performance_data['dataparallel'] = dp_time
    print(f"DataParallel training time: {dp_time:.6f} seconds per iteration")
    
    # Test with balanced DataParallel
    print("\nTesting balanced DataParallel...")
    model_bdp = get_balanced_data_parallel(model)
    bdp_time = test_training_performance(model_bdp, data, target, args.iterations)
    performance_data['balanced_dataparallel'] = bdp_time
    print(f"Balanced DataParallel training time: {bdp_time:.6f} seconds per iteration")
    
    # Test with model parallelism
    print("\nTesting ModelParallelModule...")
    model_mp = ModelParallelModule(model)
    mp_time = test_training_performance(model_mp, data, target, args.iterations)
    performance_data['model_parallel'] = mp_time
    print(f"Model parallel training time: {mp_time:.6f} seconds per iteration")
    
    # Test with pipeline parallelism
    print("\nTesting PipelineParallelModule...")
    model_pp = PipelineParallelModule(model, num_microbatches=4)
    pp_time = test_training_performance(model_pp, data, target, args.iterations)
    performance_data['pipeline_parallel'] = pp_time
    print(f"Pipeline parallel training time: {pp_time:.6f} seconds per iteration")
    
    # Test with auto parallelism
    print("\nTesting auto_parallelize_model...")
    model_auto = auto_parallelize_model(model, model_size_mb=500)
    auto_time = test_training_performance(model_auto, data, target, args.iterations)
    performance_data['auto_parallel'] = auto_time
    print(f"Auto parallel training time: {auto_time:.6f} seconds per iteration")
    
    # Plot results
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    labels = list(performance_data.keys())
    times = list(performance_data.values())
    speedups = [baseline_time / t for t in times]
    
    plt.barh(labels, speedups, color='skyblue')
    plt.xlabel('Speedup vs Baseline')
    plt.title('Model Parallelism Performance Comparison')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text to bars
    for i, (label, speedup, time) in enumerate(zip(labels, speedups, times)):
        plt.text(speedup + 0.05, i, f"{speedup:.2f}x ({time:.4f}s)", va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_parallelism_comparison.png'))
    print(f"\nResults saved to {os.path.join(args.output_dir, 'model_parallelism_comparison.png')}")

def run_benchmarks(args):
    """Run GPU benchmarks"""
    print("\n=== Running GPU Benchmarks ===\n")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print GPU information
    print("\nGPU Information:")
    print_gpu_info()
    
    # Apply cuDNN optimizations
    print("\nApplying CUDA/cuDNN optimizations:")
    optimize_cudnn()
    
    # Matrix multiplication benchmark
    print("\nRunning matrix multiplication benchmark:")
    matmul_results = CUDABenchmark.benchmark_matmul(
        sizes=[(1000, 1000), (2000, 2000), (4000, 4000)],
        iterations=5
    )
    
    # Memory transfer benchmark
    print("\nRunning memory transfer benchmark:")
    transfer_results = CUDABenchmark.benchmark_memory_transfer(
        sizes=[10, 100, 500],
        iterations=5
    )
    
    # Plot matrix multiplication results
    plt.figure(figsize=(10, 6))
    sizes = [f"{size[0]}x{size[1]}" for size in matmul_results.keys()]
    gflops = [result['gflops'] for result in matmul_results.values()]
    
    plt.bar(sizes, gflops, color='skyblue')
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Matrix Multiplication Performance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, (size, gflop) in enumerate(zip(sizes, gflops)):
        plt.text(i, gflop + 10, f"{gflop:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'matmul_benchmark.png'))
    
    # Plot memory transfer results
    plt.figure(figsize=(10, 8))
    sizes = list(transfer_results.keys())
    
    # CPU to GPU
    plt.subplot(2, 1, 1)
    bandwidths = [result['to_gpu_bandwidth'] for result in transfer_results.values()]
    plt.bar([str(s) for s in sizes], bandwidths, color='skyblue')
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('CPU to GPU Transfer Speed')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # GPU to CPU
    plt.subplot(2, 1, 2)
    bandwidths = [result['to_cpu_bandwidth'] for result in transfer_results.values()]
    plt.bar([str(s) for s in sizes], bandwidths, color='lightgreen')
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('GPU to CPU Transfer Speed')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'memory_transfer_benchmark.png'))
    
    print(f"\nBenchmark results saved to {args.output_dir}")

def compare_training_performance(args):
    """Compare training performance between original and optimized versions"""
    print("\n=== Comparing Training Performance ===\n")
    
    # Create models and data
    model_sizes = ['small', 'medium', 'large']
    batch_sizes = [32, 64, 128]
    
    results_original = {}
    results_optimized = {}
    
    for model_size in model_sizes:
        for batch_size in batch_sizes:
            print(f"\nTesting {model_size} model with batch size {batch_size}:")
            
            # Create model and data
            model = create_test_model(model_size)
            data = create_test_data(batch_size)
            target = torch.randint(0, 10, (batch_size,))
            
            # Test original (without optimizations)
            if not args.skip_original:
                print("  Testing original implementation...")
                original_time = test_training_performance(
                    model, data, target, args.iterations, use_amp=False
                )
                results_original[f"{model_size}_{batch_size}"] = original_time
                print(f"  Original training time: {original_time:.6f} seconds per iteration")
            
            # Test optimized (with AMP)
            print("  Testing optimized implementation...")
            optimized_time = test_training_performance(
                model, data, target, args.iterations, use_amp=True
            )
            results_optimized[f"{model_size}_{batch_size}"] = optimized_time
            print(f"  Optimized training time: {optimized_time:.6f} seconds per iteration")
            
            if not args.skip_original:
                speedup = original_time / optimized_time
                print(f"  Speedup: {speedup:.2f}x")
    
    # Plot results if comparison is enabled
    if args.compare and not args.skip_original:
        os.makedirs(args.output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        # Organize data for plotting
        model_batch_keys = list(results_original.keys())
        original_times = [results_original[key] for key in model_batch_keys]
        optimized_times = [results_optimized[key] for key in model_batch_keys]
        
        # Format labels
        labels = [key.replace('_', ', Batch: ') for key in model_batch_keys]
        labels = [f"Model: {label}" for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, original_times, width, label='Original', color='skyblue')
        plt.bar(x + width/2, optimized_times, width, label='Optimized', color='lightgreen')
        
        plt.ylabel('Time per iteration (seconds)')
        plt.title('Training Performance Comparison')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add speedup text
        for i, (orig, opt) in enumerate(zip(original_times, optimized_times)):
            speedup = orig / opt
            plt.text(i, max(orig, opt) + 0.01, f"{speedup:.2f}x", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_performance_comparison.png'))
        print(f"\nResults saved to {os.path.join(args.output_dir, 'training_performance_comparison.png')}")

def compare_inference_performance(args):
    """Compare inference performance between original and optimized versions"""
    print("\n=== Comparing Inference Performance ===\n")
    
    # Create models and data
    model_sizes = ['small', 'medium', 'large']
    batch_sizes = [1, 8, 32, 128]
    
    results_original = {}
    results_optimized = {}
    
    for model_size in model_sizes:
        for batch_size in batch_sizes:
            print(f"\nTesting {model_size} model with batch size {batch_size}:")
            
            # Create model and data
            model = create_test_model(model_size)
            data = create_test_data(batch_size)
            
            # Test original (without optimizations)
            if not args.skip_original:
                print("  Testing original implementation...")
                original_time = test_inference_performance(
                    model, data, args.iterations, use_amp=False
                )
                results_original[f"{model_size}_{batch_size}"] = original_time
                print(f"  Original inference time: {original_time:.6f} seconds per iteration")
            
            # Test optimized (with AMP)
            print("  Testing optimized implementation...")
            optimized_time = test_inference_performance(
                model, data, args.iterations, use_amp=True
            )
            results_optimized[f"{model_size}_{batch_size}"] = optimized_time
            print(f"  Optimized inference time: {optimized_time:.6f} seconds per iteration")
            
            if not args.skip_original:
                speedup = original_time / optimized_time
                print(f"  Speedup: {speedup:.2f}x")
    
    # Plot results if comparison is enabled
    if args.compare and not args.skip_original:
        os.makedirs(args.output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        # Organize data for plotting
        model_batch_keys = list(results_original.keys())
        original_times = [results_original[key] for key in model_batch_keys]
        optimized_times = [results_optimized[key] for key in model_batch_keys]
        
        # Format labels
        labels = [key.replace('_', ', Batch: ') for key in model_batch_keys]
        labels = [f"Model: {label}" for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, original_times, width, label='Original', color='skyblue')
        plt.bar(x + width/2, optimized_times, width, label='Optimized', color='lightgreen')
        
        plt.ylabel('Time per iteration (seconds)')
        plt.title('Inference Performance Comparison')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add speedup text
        for i, (orig, opt) in enumerate(zip(original_times, optimized_times)):
            speedup = orig / opt
            plt.text(i, max(orig, opt) + 0.01, f"{speedup:.2f}x", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'inference_performance_comparison.png'))
        print(f"\nResults saved to {os.path.join(args.output_dir, 'inference_performance_comparison.png')}")

def main():
    """Main function"""
    args = setup_args()
    
    print("=" * 50)
    print("CUDA Performance Testing Script")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Results will be CPU-only.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected tests
    if args.mode == 'benchmark' or args.mode == 'both':
        run_benchmarks(args)
    
    if args.mode == 'train' or args.mode == 'both':
        compare_training_performance(args)
    
    if args.mode == 'inference' or args.mode == 'both':
        compare_inference_performance(args)
    
    if args.model_test:
        test_model_parallelism(args)
    
    print("\nAll performance tests completed!")

if __name__ == "__main__":
    main() 