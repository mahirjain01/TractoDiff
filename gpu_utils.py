import os
import time
import threading
import subprocess
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from datetime import datetime

class GPUStats:
    """Class to monitor and log GPU statistics during training/inference"""
    
    def __init__(self, log_dir='gpu_logs', interval=1.0, devices=None):
        """
        Initialize GPU monitoring
        
        Args:
            log_dir: Directory to store GPU monitoring logs
            interval: Sampling interval in seconds
            devices: List of GPU device IDs to monitor (None = all available)
        """
        self.log_dir = log_dir
        self.interval = interval
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set devices to monitor
        if devices is None:
            self.devices = list(range(torch.cuda.device_count()))
        else:
            self.devices = devices
            
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {device: {'util': [], 'mem': [], 'temp': [], 'time': []} for device in self.devices}
        
        # Create CSV log files for each device
        self.csv_files = {}
        self.csv_writers = {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for device in self.devices:
            filename = os.path.join(log_dir, f"gpu_{device}_stats_{timestamp}.csv")
            self.csv_files[device] = open(filename, 'w', newline='')
            self.csv_writers[device] = csv.writer(self.csv_files[device])
            # Write header
            self.csv_writers[device].writerow(['timestamp', 'utilization_gpu', 'memory_used_mib', 'temperature_gpu'])
    
    def _get_gpu_stats(self):
        """Get current GPU stats using nvidia-smi"""
        try:
            # Run nvidia-smi to get GPU stats
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            stats = {}
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    device_id = int(parts[0])
                    if device_id in self.devices:
                        stats[device_id] = {
                            'util': float(parts[1]),
                            'mem': float(parts[2]),
                            'temp': float(parts[3])
                        }
            
            return stats
        except (subprocess.SubprocessError, ValueError) as e:
            print(f"Error getting GPU stats: {e}")
            return {}
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        
        while self.monitoring:
            # Get current stats
            current_time = time.time() - start_time
            try:
                gpu_stats = self._get_gpu_stats()
                
                # Store stats
                for device in self.devices:
                    if device in gpu_stats:
                        self.stats[device]['time'].append(current_time)
                        self.stats[device]['util'].append(gpu_stats[device]['util'])
                        self.stats[device]['mem'].append(gpu_stats[device]['mem'])
                        self.stats[device]['temp'].append(gpu_stats[device]['temp'])
                        
                        # Write to CSV
                        self.csv_writers[device].writerow([
                            current_time,
                            gpu_stats[device]['util'],
                            gpu_stats[device]['mem'],
                            gpu_stats[device]['temp']
                        ])
                        self.csv_files[device].flush()  # Ensure data is written immediately
            except Exception as e:
                print(f"Error in GPU monitoring: {e}")
            
            # Sleep until next sample
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        if self.monitoring:
            print("GPU monitoring is already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"GPU monitoring started for devices {self.devices}")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        if not self.monitoring:
            print("GPU monitoring is not running")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Close CSV files
        for device in self.devices:
            if device in self.csv_files:
                self.csv_files[device].close()
        
        print(f"GPU monitoring stopped")
    
    def plot_stats(self, save_dir=None):
        """Plot collected GPU statistics"""
        if not save_dir:
            save_dir = self.log_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot each device
        for device in self.devices:
            if len(self.stats[device]['time']) == 0:
                continue
                
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Utilization
            ax1.plot(self.stats[device]['time'], self.stats[device]['util'])
            ax1.set_ylabel('GPU Utilization (%)')
            ax1.set_title(f'GPU {device} Statistics')
            ax1.grid(True)
            
            # Memory
            ax2.plot(self.stats[device]['time'], self.stats[device]['mem'])
            ax2.set_ylabel('Memory Usage (MiB)')
            ax2.grid(True)
            
            # Temperature
            ax3.plot(self.stats[device]['time'], self.stats[device]['temp'])
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Temperature (°C)')
            ax3.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(save_dir, f"gpu_{device}_plot_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close(fig)
            
            print(f"Saved GPU statistics plot to {plot_path}")
            
        # Combined plot for all devices
        if len(self.devices) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            # Utilization for all devices
            for device in self.devices:
                if len(self.stats[device]['time']) > 0:
                    ax1.plot(self.stats[device]['time'], self.stats[device]['util'], label=f'GPU {device}')
            
            ax1.set_ylabel('GPU Utilization (%)')
            ax1.set_title('GPU Utilization Comparison')
            ax1.grid(True)
            ax1.legend()
            
            # Memory for all devices
            for device in self.devices:
                if len(self.stats[device]['time']) > 0:
                    ax2.plot(self.stats[device]['time'], self.stats[device]['mem'], label=f'GPU {device}')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Memory Usage (MiB)')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save combined plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(save_dir, f"gpu_combined_plot_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close(fig)
            
            print(f"Saved combined GPU statistics plot to {plot_path}")
    
    def get_summary(self):
        """Get summary statistics for all monitored GPUs"""
        summary = {}
        
        for device in self.devices:
            if len(self.stats[device]['util']) == 0:
                continue
                
            summary[device] = {
                'util_mean': np.mean(self.stats[device]['util']),
                'util_max': np.max(self.stats[device]['util']),
                'mem_mean': np.mean(self.stats[device]['mem']),
                'mem_max': np.max(self.stats[device]['mem']),
                'temp_mean': np.mean(self.stats[device]['temp']),
                'temp_max': np.max(self.stats[device]['temp']),
                'duration': self.stats[device]['time'][-1] if self.stats[device]['time'] else 0
            }
        
        return summary

class CUDABenchmark:
    """Benchmarking utilities for CUDA operations"""
    
    @staticmethod
    def benchmark_matmul(sizes=[(1000, 1000), (2000, 2000), (4000, 4000)], 
                         dtype=torch.float32, device=None, iterations=10):
        """
        Benchmark matrix multiplication performance
        
        Args:
            sizes: List of (m, n) tuples for matrix sizes
            dtype: Tensor data type
            device: Device to run on (None = default CUDA device)
            iterations: Number of iterations to average over
        
        Returns:
            Dictionary of results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        results = {}
        
        for size in sizes:
            m, n = size
            
            # Create random matrices
            a = torch.randn(m, n, dtype=dtype, device=device)
            b = torch.randn(n, m, dtype=dtype, device=device)
            
            # Warmup
            for _ in range(5):
                c = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            gflops = (2 * m * n * m) / (avg_time * 1e9)  # Giga FLOPS
            
            results[size] = {
                'avg_time': avg_time,
                'gflops': gflops
            }
            
            print(f"Matrix multiplication {m}x{n} x {n}x{m}: {avg_time:.6f} s, {gflops:.2f} GFLOPS")
        
        return results
    
    @staticmethod
    def benchmark_memory_transfer(sizes=[1, 10, 100, 1000], 
                                  dtype=torch.float32, iterations=10):
        """
        Benchmark CPU to GPU memory transfer speeds
        
        Args:
            sizes: List of tensor sizes in MB
            dtype: Tensor data type
            iterations: Number of iterations to average over
        
        Returns:
            Dictionary of results
        """
        if not torch.cuda.is_available():
            print("CUDA is not available")
            return {}
        
        results = {}
        
        for size_mb in sizes:
            # Calculate number of elements
            bytes_per_element = torch.tensor([], dtype=dtype).element_size()
            elements = int(size_mb * 1024 * 1024 / bytes_per_element)
            
            # Create tensor on CPU
            cpu_tensor = torch.randn(elements, dtype=dtype)
            
            # Warmup
            for _ in range(5):
                gpu_tensor = cpu_tensor.cuda()
                torch.cuda.synchronize()
                del gpu_tensor
            
            # Benchmark CPU to GPU
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(iterations):
                gpu_tensor = cpu_tensor.cuda()
                torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time_to_gpu = (end_time - start_time) / iterations
            bandwidth_to_gpu = size_mb / avg_time_to_gpu
            
            # Benchmark GPU to CPU
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(iterations):
                cpu_tensor_copy = gpu_tensor.cpu()
                torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time_to_cpu = (end_time - start_time) / iterations
            bandwidth_to_cpu = size_mb / avg_time_to_cpu
            
            results[size_mb] = {
                'to_gpu_time': avg_time_to_gpu,
                'to_gpu_bandwidth': bandwidth_to_gpu,
                'to_cpu_time': avg_time_to_cpu,
                'to_cpu_bandwidth': bandwidth_to_cpu
            }
            
            print(f"Memory transfer {size_mb} MB:")
            print(f"  CPU -> GPU: {avg_time_to_gpu:.6f} s, {bandwidth_to_gpu:.2f} MB/s")
            print(f"  GPU -> CPU: {avg_time_to_cpu:.6f} s, {bandwidth_to_cpu:.2f} MB/s")
        
        return results

def optimize_cudnn():
    """Apply optimal cuDNN settings for the current hardware"""
    if not torch.cuda.is_available():
        print("CUDA is not available. cuDNN optimizations skipped.")
        return False
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Ensure deterministic algorithms are disabled for speed
    torch.backends.cudnn.deterministic = False
    
    # Allow TF32 format on Ampere or later GPUs
    if hasattr(torch.backends.cuda, 'matmul'):
        # Enable TF32 for matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        # Enable TF32 for convolutions
        torch.backends.cudnn.allow_tf32 = True
    
    print("CUDA/cuDNN optimizations applied:")
    print(f"  cudnn.benchmark = {torch.backends.cudnn.benchmark}")
    print(f"  cudnn.deterministic = {torch.backends.cudnn.deterministic}")
    if hasattr(torch.backends.cuda, 'matmul'):
        print(f"  cuda.matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        print(f"  cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}")
    
    return True

def get_gpu_memory_usage():
    """
    Get current GPU memory usage
    
    Returns:
        Dictionary mapping device IDs to (used, total) memory in MiB
    """
    if not torch.cuda.is_available():
        return {}
        
    result = {}
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024 ** 2)  # MiB
        
        # Get current memory usage
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MiB
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MiB
        
        result[i] = {
            'allocated': allocated,
            'reserved': reserved,
            'total': total_memory,
            'free': total_memory - reserved
        }
    
    return result

def print_gpu_info():
    """Print detailed information about available GPUs"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    if hasattr(torch.backends, 'cudnn'):
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    print(f"\nFound {torch.cuda.device_count()} CUDA device(s):")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")
        print(f"  Max threads per block: {props.max_threads_per_block}")
        print(f"  Max threads per multi-processor: {props.max_threads_per_multi_processor}")
        
        # Current memory usage
        mem_info = get_gpu_memory_usage()[i]
        print(f"  Current memory allocated: {mem_info['allocated']:.2f} MiB")
        print(f"  Current memory reserved: {mem_info['reserved']:.2f} MiB")
        print(f"  Current memory free: {mem_info['free']:.2f} MiB")

if __name__ == "__main__":
    # Example usage
    print_gpu_info()
    optimize_cudnn()
    
    # Run a quick benchmark
    bench = CUDABenchmark()
    bench.benchmark_matmul(sizes=[(1000, 1000), (2000, 2000)])
    bench.benchmark_memory_transfer(sizes=[10, 100]) 