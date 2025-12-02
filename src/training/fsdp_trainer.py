"""FSDP (Fully Sharded Data Parallel) trainer implementation."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from typing import Dict, Any

from .base_trainer import BaseTrainer
from ..utils.logging import setup_logging


def get_fsdp_wrap_policy(model_class):
    """
    Get auto-wrap policy for FSDP.
    
    Wraps transformer blocks individually for better memory efficiency.
    """
    def transformer_wrap_policy(module, recurse, unwrapped_params):
        if recurse:
            return True
        return isinstance(module, model_class)
    
    return transformer_wrap_policy


class FSDPTrainer(BaseTrainer):
    """
    Trainer using PyTorch Fully Sharded Data Parallel (FSDP).
    
    FSDP shards model parameters, gradients, and optimizer states across GPUs.
    Best for large models that don't fit on a single GPU.
    
    Key characteristics:
    - Model parameters sharded across GPUs
    - Gradients and optimizer states also sharded
    - Higher memory efficiency (no full model copy)
    - Higher communication overhead (all-gather during forward/backward)
    - Best for: Large models that exceed single GPU memory
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize FSDP trainer.
        
        Args:
            model: PyTorch model
            config: Training configuration
            device: Device to train on
            rank: Process rank
            world_size: Total number of processes
            sharding_strategy: FSDP sharding strategy
            use_mixed_precision: Whether to use mixed precision training
        """
        self.sharding_strategy = sharding_strategy
        self.use_mixed_precision = use_mixed_precision
        super().__init__(model, config, device, rank, world_size)
        self.setup_distributed()
    
    def setup_distributed(self) -> None:
        """Setup FSDP model wrapping."""
        if self.world_size > 1:
            # Import transformer block for auto-wrapping
            from ..models.simple_model import TransformerBlock
            
            # Auto-wrap policy: wrap each TransformerBlock
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerBlock},
            )
            
            # Mixed precision config
            mixed_precision_policy = None
            if self.use_mixed_precision:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float32,
                )
            
            # Wrap model in FSDP
            self.model = FSDP(
                self.model,
                sharding_strategy=self.sharding_strategy,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                device_id=self.device,
            )
            
            if self.rank == 0:
                print(f"FSDP initialized: {self.world_size} GPUs")
                print(f"Sharding strategy: {self.sharding_strategy}")
                print(f"Mixed precision: {self.use_mixed_precision}")
        else:
            if self.rank == 0:
                print("Single GPU mode (FSDP not needed)")
    
    def train_step(self) -> Dict[str, float]:
        """
        Execute a single training step with FSDP.
        
        Returns:
            Dictionary with step metrics
        """
        # FSDP handles gradient synchronization automatically
        metrics = super().train_step()
        
        if self.world_size > 1:
            # Average loss across all processes
            dist.all_reduce(torch.tensor(metrics["loss"], device=self.device), op=dist.ReduceOp.SUM)
            metrics["loss"] = metrics["loss"] / self.world_size
        
        return metrics

