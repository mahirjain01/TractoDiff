import torch
import torch.nn as nn
import numpy as np
import warnings

class DeviceBalancer:
    """
    Utility to balance workload across heterogeneous GPUs based on their memory capacity.
    This handles the case where we have GPUs with different memory sizes.
    """
    
    def __init__(self):
        """Initialize the device balancer"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Using CPU only.")
            self.devices = ["cpu"]
            self.memory_ratios = [1.0]
            return
            
        # Get number of available GPUs
        self.num_devices = torch.cuda.device_count()
        self.devices = []
        
        # Get memory for each device
        self.memory_sizes = []
        
        # Collect all GPU information
        for i in range(self.num_devices):
            device = f"cuda:{i}"
            self.devices.append(device)
            
            # Get memory size
            props = torch.cuda.get_device_properties(i)
            self.memory_sizes.append(props.total_memory)
        
        # Calculate ratios for workload distribution
        total_memory = sum(self.memory_sizes)
        self.memory_ratios = [size / total_memory for size in self.memory_sizes]
        
        print(f"DeviceBalancer initialized with {self.num_devices} GPUs")
        for i, (device, memory, ratio) in enumerate(zip(self.devices, self.memory_sizes, self.memory_ratios)):
            memory_gb = memory / (1024**3)
            print(f"  GPU {i}: {memory_gb:.2f} GB ({ratio:.2%} of total)")
    
    def get_optimal_device(self, model_size_mb=None):
        """
        Returns the optimal device to use based on available memory
        
        Args:
            model_size_mb: Approximate model size in MB (if known)
        
        Returns:
            The device with the most available memory
        """
        if not torch.cuda.is_available():
            return "cpu"
            
        # If model size is small or unknown, just use the device with most total memory
        if model_size_mb is None or model_size_mb < 1000:  # Less than 1GB
            return self.devices[np.argmax(self.memory_sizes)]
        
        # Check current available memory
        available_memory = []
        for i in range(self.num_devices):
            # Get current memory usage
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            
            # Calculate available memory
            available = total - reserved
            available_memory.append(available)
        
        # Convert model size to bytes
        model_size_bytes = model_size_mb * 1024 * 1024
        
        # Check if model fits on any GPU
        for i, available in enumerate(available_memory):
            if available >= model_size_bytes * 1.2:  # Add 20% buffer
                return self.devices[i]
        
        # If we get here, the model might be too large for any single GPU
        warnings.warn("Model may be too large for any single GPU. Using the device with most available memory.")
        return self.devices[np.argmax(available_memory)]
    
    def get_device_for_batch(self, batch_idx, num_batches):
        """
        Distribute batches across devices based on memory ratios
        
        Args:
            batch_idx: Current batch index
            num_batches: Total number of batches
            
        Returns:
            Device to use for this batch
        """
        if not torch.cuda.is_available() or self.num_devices == 1:
            return self.devices[0]
        
        # Divide batches according to memory ratios
        thresholds = [0]
        cumulative = 0
        for ratio in self.memory_ratios[:-1]:  # Don't need the last one
            cumulative += ratio
            thresholds.append(cumulative)
        
        # Normalize batch index to [0, 1] range
        position = batch_idx / num_batches
        
        # Find the right device based on position
        for i, threshold in enumerate(thresholds[1:], 1):
            if position < threshold:
                return self.devices[i-1]
        
        return self.devices[-1]  # Last device

class ModelParallelModule(nn.Module):
    """
    Base class for implementing model parallelism across heterogeneous GPUs.
    This splits large models across multiple GPUs based on layers/blocks.
    """
    
    def __init__(self, base_model, devices=None):
        """
        Initialize a model parallel module
        
        Args:
            base_model: The model to parallelize
            devices: List of devices to use (if None, all available GPUs will be used)
        """
        super().__init__()
        
        if devices is None:
            # Use all available GPUs
            self.num_gpus = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)]
        else:
            self.devices = devices
            self.num_gpus = len(devices)
        
        if self.num_gpus <= 1:
            # No need for model parallelism
            self.model = base_model
            if torch.cuda.is_available():
                self.model = self.model.to("cuda:0")
            return
            
        # Split the model into chunks based on number of GPUs
        self.parallel_model = self._parallelize_model(base_model)
    
    def _parallelize_model(self, model):
        """
        Split the model across multiple GPUs
        This is a simple implementation that splits sequential models by layer groups
        More complex models would need custom splitting logic
        
        Args:
            model: The model to parallelize
            
        Returns:
            Dictionary mapping devices to model parts
        """
        if not hasattr(model, "children"):
            # Can't split, just return the model on the first device
            return {self.devices[0]: model.to(self.devices[0])}
            
        # Get list of children
        children = list(model.children())
        
        if not children:
            # No children, can't split
            return {self.devices[0]: model.to(self.devices[0])}
        
        # Split children across devices
        chunks = {}
        layers_per_gpu = max(1, len(children) // self.num_gpus)
        
        for i, device in enumerate(self.devices):
            start_idx = i * layers_per_gpu
            end_idx = min((i + 1) * layers_per_gpu, len(children))
            
            if start_idx >= len(children):
                break
                
            # Create a sequential model from these layers
            chunk = nn.Sequential(*children[start_idx:end_idx]).to(device)
            chunks[device] = chunk
            
        return chunks
    
    def forward(self, x):
        """
        Forward pass through the parallel model
        
        Args:
            x: Input tensor (assumed to be on CPU or the first device)
            
        Returns:
            Output tensor (on the last device used)
        """
        if self.num_gpus <= 1:
            # Single GPU or CPU mode
            if torch.cuda.is_available():
                x = x.to("cuda:0")
            return self.model(x)
        
        # Multi-GPU mode
        current_device = None
        
        for device, module in self.parallel_model.items():
            # Move input to the current device if needed
            if current_device != device:
                x = x.to(device)
                current_device = device
            
            # Run this part of the model
            x = module(x)
        
        return x

class PipelineParallelModule(nn.Module):
    """
    Implementation of pipeline parallelism for heterogeneous GPUs.
    Splits the model into stages and processes different microbatches in parallel.
    """
    
    def __init__(self, base_model, num_microbatches=4, devices=None):
        """
        Initialize a pipeline parallel module
        
        Args:
            base_model: The model to parallelize
            num_microbatches: Number of microbatches to use
            devices: List of devices to use (if None, all available GPUs will be used)
        """
        super().__init__()
        
        if devices is None:
            # Use all available GPUs
            self.num_gpus = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)]
        else:
            self.devices = devices
            self.num_gpus = len(devices)
        
        self.num_microbatches = num_microbatches
        
        if self.num_gpus <= 1:
            # No need for pipeline parallelism
            self.model = base_model
            if torch.cuda.is_available():
                self.model = self.model.to("cuda:0")
            self.is_parallel = False
            return
            
        # Split the model into stages
        self.stages = self._create_pipeline_stages(base_model)
        self.is_parallel = True
    
    def _create_pipeline_stages(self, model):
        """
        Split the model into pipeline stages
        
        Args:
            model: The model to parallelize
            
        Returns:
            List of pipeline stages
        """
        if not hasattr(model, "children"):
            # Can't split, just return the model on the first device
            return [model.to(self.devices[0])]
            
        # Get list of children
        children = list(model.children())
        
        if not children:
            # No children, can't split
            return [model.to(self.devices[0])]
        
        # Split children across devices
        stages = []
        layers_per_stage = max(1, len(children) // self.num_gpus)
        
        for i in range(self.num_gpus):
            start_idx = i * layers_per_stage
            end_idx = min((i + 1) * layers_per_stage, len(children))
            
            if start_idx >= len(children):
                break
                
            # Create a sequential model from these layers
            stage = nn.Sequential(*children[start_idx:end_idx]).to(self.devices[i])
            stages.append(stage)
            
        return stages
    
    def _process_microbatch(self, x, stage_idx=0):
        """
        Process a single microbatch through a specific stage
        
        Args:
            x: Input tensor
            stage_idx: Stage index to process
            
        Returns:
            Output tensor from this stage
        """
        device = self.devices[stage_idx]
        x = x.to(device)
        return self.stages[stage_idx](x)
    
    def forward(self, x):
        """
        Forward pass using pipeline parallelism
        
        Args:
            x: Input tensor (batch)
            
        Returns:
            Output tensor
        """
        if not self.is_parallel:
            # Single GPU or CPU mode
            device = self.devices[0] if torch.cuda.is_available() else "cpu"
            x = x.to(device)
            return self.model(x)
        
        # Split input into microbatches
        batch_size = x.size(0)
        microbatch_size = max(1, batch_size // self.num_microbatches)
        microbatches = []
        
        for i in range(0, batch_size, microbatch_size):
            end_idx = min(i + microbatch_size, batch_size)
            microbatches.append(x[i:end_idx])
        
        # Pipeline processing
        outputs = []
        
        for mb_idx, mb in enumerate(microbatches):
            # Process this microbatch through all stages
            current_output = mb
            
            for stage_idx in range(len(self.stages)):
                current_output = self._process_microbatch(current_output, stage_idx)
            
            outputs.append(current_output)
        
        # Combine microbatch outputs
        return torch.cat(outputs, dim=0)

def auto_parallelize_model(model, model_size_mb=None, use_pipeline=False, num_microbatches=None):
    """
    Automatically parallelize a model based on available resources
    
    Args:
        model: The model to parallelize
        model_size_mb: Estimated model size in MB (if known)
        use_pipeline: Whether to use pipeline parallelism
        num_microbatches: Number of microbatches for pipeline parallelism
        
    Returns:
        Parallelized model
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return model.to("cpu")
    
    # Check GPU count
    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        # Only one GPU, no need for model parallelism
        print("Only one GPU available. Using simple GPU acceleration.")
        return model.to("cuda:0")
    
    # Get memory for each device
    memory_sizes = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_sizes.append(props.total_memory)
    
    # Check if we have heterogeneous GPUs
    if len(set(memory_sizes)) > 1:
        print(f"Detected {num_gpus} GPUs with heterogeneous memory sizes.")
        
        # If model size is known and it's large, use model parallelism
        if model_size_mb and model_size_mb > min(memory_sizes) / (1024 * 1024):
            if use_pipeline:
                # Use pipeline parallelism
                if num_microbatches is None:
                    # Default to 2x number of GPUs
                    num_microbatches = num_gpus * 2
                
                print(f"Using pipeline parallelism with {num_microbatches} microbatches.")
                return PipelineParallelModule(model, num_microbatches=num_microbatches)
            else:
                # Use model parallelism
                print("Using model parallelism due to large model size.")
                return ModelParallelModule(model)
    
    # Default: Use the GPU with the most memory
    best_gpu = memory_sizes.index(max(memory_sizes))
    print(f"Using GPU {best_gpu} with {memory_sizes[best_gpu] / (1024**3):.2f} GB memory.")
    return model.to(f"cuda:{best_gpu}")

def get_balanced_data_parallel(model):
    """
    Create a DataParallel model with balanced load across GPUs
    
    Args:
        model: Base model
        
    Returns:
        DataParallel model with adjusted chunk sizes
    """
    if not torch.cuda.is_available():
        return model.to("cpu")
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        return model.to("cuda:0")
    
    # Get memory ratio for each device
    memory_sizes = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_sizes.append(props.total_memory)
    
    total_memory = sum(memory_sizes)
    memory_ratios = [size / total_memory for size in memory_sizes]
    
    # Calculate chunk sizes proportional to memory sizes
    # PyTorch uses equal chunks by default, but we can balance based on GPU capabilities
    chunk_sizes = [round(ratio * 100) for ratio in memory_ratios]
    
    # Create custom DataParallel with balanced chunk sizes
    class BalancedDataParallel(torch.nn.DataParallel):
        def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
            super().__init__(module, device_ids, output_device, dim)
            self.chunk_sizes = chunk_sizes
        
        def scatter(self, inputs, kwargs, device_ids):
            if self.chunk_sizes:
                # Customize chunking based on memory ratios
                return self._custom_scatter(inputs, kwargs, device_ids, self.chunk_sizes)
            return super().scatter(inputs, kwargs, device_ids)
        
        def _custom_scatter(self, inputs, kwargs, device_ids, chunk_sizes):
            # Only handle simple case of single input tensor
            if len(inputs) == 1 and isinstance(inputs[0], torch.Tensor):
                batch_size = inputs[0].size(0)
                
                # Convert chunk sizes to actual number of samples
                sizes = [max(1, round(c * batch_size / sum(chunk_sizes))) for c in chunk_sizes[:len(device_ids)]]
                
                # Adjust to ensure total equals batch size
                diff = batch_size - sum(sizes)
                sizes[-1] += diff
                
                # Create chunks
                chunks = []
                start = 0
                for s in sizes:
                    chunks.append(inputs[0][start:start+s])
                    start += s
                
                # Move chunks to devices
                scattered = [(chunk.to(device_ids[i]),) for i, chunk in enumerate(chunks)]
                
                # Handle kwargs
                scattered_kwargs = [{} for _ in range(len(device_ids))]
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        # Split kwargs tensors too if they match batch dimension
                        if v.size(0) == batch_size:
                            v_chunks = []
                            start = 0
                            for s in sizes:
                                v_chunks.append(v[start:start+s])
                                start += s
                            for i, v_chunk in enumerate(v_chunks):
                                scattered_kwargs[i][k] = v_chunk.to(device_ids[i])
                        else:
                            for i in range(len(scattered_kwargs)):
                                scattered_kwargs[i][k] = v.to(device_ids[i])
                    else:
                        for i in range(len(scattered_kwargs)):
                            scattered_kwargs[i][k] = v
                
                return scattered, scattered_kwargs
            
            # Fall back to default behavior for other cases
            return super().scatter(inputs, kwargs, device_ids)
    
    model = model.to("cuda:0")
    device_ids = list(range(num_gpus))
    
    # Return balanced data parallel model
    return BalancedDataParallel(model, device_ids=device_ids, chunk_sizes=chunk_sizes)

if __name__ == "__main__":
    # Test the utilities
    def print_separator():
        print("\n" + "="*50 + "\n")
    
    print_separator()
    print("Testing DeviceBalancer:")
    balancer = DeviceBalancer()
    
    print_separator()
    print("Testing auto_parallelize_model with a test model:")
    
    # Create a test model
    test_model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    )
    
    # Try different parallelization methods
    print("\nTesting auto_parallelize_model:")
    parallel_model = auto_parallelize_model(test_model, model_size_mb=10)
    
    print("\nTesting get_balanced_data_parallel:")
    data_parallel_model = get_balanced_data_parallel(test_model)
    
    print_separator()
    print("Tests completed successfully.") 