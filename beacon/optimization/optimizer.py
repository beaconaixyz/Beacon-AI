import torch
from typing import Dict, Any, Optional, Union
from ..core.base import BeaconBase

class Optimizer(BeaconBase):
    """Base class for optimization tools"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'batch_size_range': [32, 64, 128, 256],
            'max_memory_gb': 4.0,
            'use_mixed_precision': True,
            'enable_cuda_graphs': False,
            'num_workers': 4
        }
    
    def optimize_batch_size(self, 
                          model: torch.nn.Module,
                          sample_batch: Dict[str, torch.Tensor]) -> int:
        """Find optimal batch size"""
        best_batch_size = self.config['batch_size_range'][0]
        best_throughput = 0
        
        for batch_size in self.config['batch_size_range']:
            try:
                throughput = self._measure_throughput(model, sample_batch, batch_size)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
            except RuntimeError:  # Out of memory
                break
                
        return best_batch_size
    
    def enable_mixed_precision(self, 
                             model: torch.nn.Module) -> torch.nn.Module:
        """Enable mixed precision training"""
        if self.config['use_mixed_precision'] and self.device.type == 'cuda':
            model = model.half()
        return model
    
    def optimize_memory_usage(self, 
                            data: Dict[str, torch.Tensor],
                            max_memory_gb: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Optimize memory usage of tensors"""
        if max_memory_gb is None:
            max_memory_gb = self.config['max_memory_gb']
            
        optimized_data = {}
        for key, tensor in data.items():
            if self._should_use_half_precision(tensor):
                tensor = tensor.half()
            optimized_data[key] = tensor
            
        return optimized_data
    
    def _measure_throughput(self,
                          model: torch.nn.Module,
                          sample_batch: Dict[str, torch.Tensor],
                          batch_size: int) -> float:
        """Measure model throughput"""
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # Warm up
            for _ in range(3):
                _ = model(self._adjust_batch_size(sample_batch, batch_size))
            
            start_time.record()
            for _ in range(10):
                _ = model(self._adjust_batch_size(sample_batch, batch_size))
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
            return (10 * batch_size) / (elapsed_time / 1000)  # samples per second
    
    def _should_use_half_precision(self, tensor: torch.Tensor) -> bool:
        """Determine if tensor should use half precision"""
        if not tensor.is_floating_point():
            return False
        if tensor.shape[0] < 1000:  # Small tensors
            return False
        return True
    
    def _adjust_batch_size(self,
                          batch: Dict[str, torch.Tensor],
                          target_size: int) -> Dict[str, torch.Tensor]:
        """Adjust batch size of input tensors"""
        adjusted_batch = {}
        current_size = next(iter(batch.values())).shape[0]
        
        for key, tensor in batch.items():
            if current_size != target_size:
                if target_size > current_size:
                    # Repeat tensors to reach target size
                    repeats = (target_size + current_size - 1) // current_size
                    tensor = tensor.repeat(repeats, *[1] * (len(tensor.shape) - 1))
                    tensor = tensor[:target_size]
                else:
                    # Slice tensors to reach target size
                    tensor = tensor[:target_size]
            adjusted_batch[key] = tensor
            
        return adjusted_batch 