"""Communication cost analysis for distributed training."""

import torch
import torch.distributed as dist
from typing import Dict, Optional
from dataclasses import dataclass, field
import time


@dataclass
class CommunicationStats:
    """Statistics about communication operations."""
    num_allreduces: int = 0
    num_allgathers: int = 0
    total_comm_time: float = 0.0  # seconds
    total_bytes_communicated: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_allreduces": self.num_allreduces,
            "num_allgathers": self.num_allgathers,
            "total_comm_time_sec": self.total_comm_time,
            "total_bytes_communicated": self.total_bytes_communicated,
            "avg_comm_time_per_op": (
                self.total_comm_time / max(1, self.num_allreduces + self.num_allgathers)
            ),
        }


class CommunicationProfiler:
    """
    Profile communication costs in distributed training.
    
    Tracks allreduce and allgather operations to understand
    communication overhead in DDP vs FSDP.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize communication profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.stats = CommunicationStats()
        self._hooks = []
        
    def _allreduce_hook(self, param: torch.Tensor) -> None:
        """Hook to track allreduce operations (DDP)."""
        if not self.enabled:
            return
        
        start_time = time.time()
        # Estimate bytes: 4 bytes per float32
        bytes_comm = param.numel() * 4 * 2  # send + receive
        
        # Note: actual allreduce happens in DDP, this is just tracking
        self.stats.num_allreduces += 1
        self.stats.total_bytes_communicated += bytes_comm
        
        # Estimate communication time (simplified)
        # Real implementation would need to measure actual comm time
        comm_time = bytes_comm / (10e9)  # Assume 10 GB/s bandwidth
        self.stats.total_comm_time += comm_time
    
    def _allgather_hook(self, param: torch.Tensor, world_size: int) -> None:
        """Hook to track allgather operations (FSDP)."""
        if not self.enabled:
            return
        
        # Estimate bytes: allgather sends shard to all ranks
        shard_size = param.numel() // world_size
        bytes_comm = shard_size * 4 * world_size  # 4 bytes per float32
        
        self.stats.num_allgathers += 1
        self.stats.total_bytes_communicated += bytes_comm
        
        # Estimate communication time
        comm_time = bytes_comm / (10e9)  # Assume 10 GB/s bandwidth
        self.stats.total_comm_time += comm_time
    
    def get_stats(self) -> CommunicationStats:
        """Get current communication statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset statistics."""
        self.stats = CommunicationStats()
    
    def estimate_ddp_comm_cost(
        self,
        model: torch.nn.Module,
        world_size: int,
    ) -> Dict:
        """
        Estimate communication cost for DDP.
        
        DDP performs allreduce on gradients after backward pass.
        Each parameter's gradient is allreduced.
        
        Args:
            model: PyTorch model
            world_size: Number of GPUs
            
        Returns:
            Dictionary with estimated communication costs
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bytes_per_allreduce = total_params * 4  # 4 bytes per float32
        
        # DDP does one allreduce per step (after backward)
        num_allreduces = 1
        
        total_bytes = bytes_per_allreduce * num_allreduces
        
        return {
            "total_parameters": total_params,
            "bytes_per_allreduce": bytes_per_allreduce,
            "estimated_comm_bytes_per_step": total_bytes,
            "estimated_comm_time_per_step": total_bytes / (10e9),  # Assume 10 GB/s
        }
    
    def estimate_fsdp_comm_cost(
        self,
        model: torch.nn.Module,
        world_size: int,
    ) -> Dict:
        """
        Estimate communication cost for FSDP.
        
        FSDP performs allgather during forward (to gather sharded params)
        and allreduce during backward (to sync gradients).
        
        Args:
            model: PyTorch model
            world_size: Number of GPUs
            
        Returns:
            Dictionary with estimated communication costs
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        shard_size = total_params // world_size
        
        # Forward: allgather sharded params
        forward_comm_bytes = shard_size * 4 * world_size  # gather from all ranks
        
        # Backward: allreduce gradients
        backward_comm_bytes = total_params * 4  # allreduce gradients
        
        total_bytes = forward_comm_bytes + backward_comm_bytes
        
        return {
            "total_parameters": total_params,
            "shard_size_per_rank": shard_size,
            "forward_comm_bytes": forward_comm_bytes,
            "backward_comm_bytes": backward_comm_bytes,
            "estimated_comm_bytes_per_step": total_bytes,
            "estimated_comm_time_per_step": total_bytes / (10e9),  # Assume 10 GB/s
        }

