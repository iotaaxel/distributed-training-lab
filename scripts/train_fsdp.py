#!/usr/bin/env python3
"""
Launch script for FSDP training.

Usage:
    # Single GPU
    python scripts/train_fsdp.py

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train_fsdp.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimpleModel
from src.training import FSDPTrainer
from src.utils.config import load_config, get_default_config
from src.utils.validation import validate_config, check_gpu_availability


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main training function."""
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "configs" / "fsdp_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = get_default_config()
            if rank == 0:
                print("Using default config (configs/fsdp_config.yaml not found)")
        
        # Validate config
        is_valid, errors = validate_config(config)
        if not is_valid:
            if rank == 0:
                print("Configuration errors:")
                for error in errors:
                    print(f"  - {error}")
            return
        
        # Check GPU availability
        if world_size > 1:
            is_available, error_msg = check_gpu_availability(world_size)
            if not is_available:
                if rank == 0:
                    print(f"GPU check failed: {error_msg}")
                return
        
        # FSDP-specific config
        sharding_strategy = config.get("fsdp", {}).get("sharding_strategy", "FULL_SHARD")
        use_mixed_precision = config.get("fsdp", {}).get("use_mixed_precision", False)
        
        # Map string to ShardingStrategy enum
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        if rank == 0:
            print("="*50)
            print("FSDP Training")
            print("="*50)
            print(f"World size: {world_size}")
            print(f"Device: {device}")
            print(f"Sharding strategy: {sharding_strategy}")
            print(f"Mixed precision: {use_mixed_precision}")
            print(f"Config: {config}")
            print()
        
        # Create model
        model = SimpleModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
        )
        
        # Create trainer
        trainer = FSDPTrainer(
            model=model,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
            sharding_strategy=sharding_strategy,
            use_mixed_precision=use_mixed_precision,
        )
        
        # Train
        summary = trainer.train(
            warmup_steps=config["benchmark"]["warmup_steps"],
            profile=False,
        )
        
        if rank == 0:
            print("\n" + "="*50)
            print("FSDP Training Complete")
            print("="*50)
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

