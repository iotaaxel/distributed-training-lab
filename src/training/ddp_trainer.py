"""DDP (DistributedDataParallel) trainer implementation."""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, Any

from .base_trainer import BaseTrainer
from ..utils.logging import setup_logging


class DDPTrainer(BaseTrainer):
    """
    Trainer using PyTorch DistributedDataParallel (DDP).
    
    DDP replicates the model on each GPU and synchronizes gradients
    during backward pass. Best for models that fit on a single GPU.
    
    Key characteristics:
    - Model replicated on each GPU
    - Gradients synchronized via allreduce
    - Lower memory efficiency (full model copy per GPU)
    - Lower communication overhead
    - Best for: Models that fit in single GPU memory
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize DDP trainer.
        
        Args:
            model: PyTorch model
            config: Training configuration
            device: Device to train on
            rank: Process rank
            world_size: Total number of processes
        """
        super().__init__(model, config, device, rank, world_size)
        self.setup_distributed()
    
    def setup_distributed(self) -> None:
        """Setup DDP model wrapping."""
        if self.world_size > 1:
            # Wrap model in DDP
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                output_device=self.device.index if self.device.type == 'cuda' else None,
                find_unused_parameters=False,  # Set to True if model has unused params
            )
            
            if self.rank == 0:
                print(f"DDP initialized: {self.world_size} GPUs")
        else:
            if self.rank == 0:
                print("Single GPU mode (DDP not needed)")
    
    def train_step(self) -> Dict[str, float]:
        """
        Execute a single training step with DDP.
        
        Returns:
            Dictionary with step metrics
        """
        # In DDP, we need to average loss across processes
        metrics = super().train_step()
        
        if self.world_size > 1:
            # Average loss across all processes
            dist.all_reduce(torch.tensor(metrics["loss"], device=self.device), op=dist.ReduceOp.SUM)
            metrics["loss"] = metrics["loss"] / self.world_size
        
        return metrics

