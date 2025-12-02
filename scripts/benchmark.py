#!/usr/bin/env python3
"""
Benchmark script to compare DDP vs FSDP performance.

Usage:
    torchrun --nproc_per_node=4 scripts/benchmark.py
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimpleModel
from src.training import DDPTrainer, FSDPTrainer
from src.utils.config import load_config, get_default_config


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
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


def benchmark_ddp(config, device, rank, world_size):
    """Benchmark DDP training."""
    if rank == 0:
        print("\n" + "="*50)
        print("Benchmarking DDP")
        print("="*50)
    
    model = SimpleModel(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )
    
    trainer = DDPTrainer(
        model=model,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    summary = trainer.train(
        warmup_steps=config["benchmark"]["warmup_steps"],
        profile=False,
    )
    
    return summary


def benchmark_fsdp(config, device, rank, world_size):
    """Benchmark FSDP training."""
    if rank == 0:
        print("\n" + "="*50)
        print("Benchmarking FSDP")
        print("="*50)
    
    from torch.distributed.fsdp import ShardingStrategy
    
    model = SimpleModel(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )
    
    sharding_strategy = config.get("fsdp", {}).get("sharding_strategy", "FULL_SHARD")
    use_mixed_precision = config.get("fsdp", {}).get("use_mixed_precision", False)
    
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    trainer = FSDPTrainer(
        model=model,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size,
        sharding_strategy=sharding_strategy,
        use_mixed_precision=use_mixed_precision,
    )
    
    summary = trainer.train(
        warmup_steps=config["benchmark"]["warmup_steps"],
        profile=False,
    )
    
    return summary


def main():
    """Main benchmarking function."""
    rank, world_size, device = setup_distributed()
    
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "configs" / "benchmark_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = get_default_config()
        
        if rank == 0:
            print("="*50)
            print("Distributed Training Benchmark")
            print("="*50)
            print(f"World size: {world_size}")
            print(f"Device: {device}")
            print()
        
        results = {}
        
        # Benchmark DDP
        ddp_summary = benchmark_ddp(config, device, rank, world_size)
        if rank == 0:
            results["ddp"] = ddp_summary
        
        # Clear CUDA cache between benchmarks
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Benchmark FSDP
        fsdp_summary = benchmark_fsdp(config, device, rank, world_size)
        if rank == 0:
            results["fsdp"] = fsdp_summary
        
        # Print comparison
        if rank == 0:
            print("\n" + "="*50)
            print("Comparison Summary")
            print("="*50)
            print(f"{'Metric':<30} {'DDP':<20} {'FSDP':<20}")
            print("-" * 70)
            
            for key in ["avg_step_time", "throughput_samples_per_sec", "peak_memory_allocated_gb"]:
                if key in ddp_summary and key in fsdp_summary:
                    ddp_val = ddp_summary[key]
                    fsdp_val = fsdp_summary[key]
                    print(f"{key:<30} {ddp_val:<20.4f} {fsdp_val:<20.4f}")
            
            # Save results
            output_path = Path(__file__).parent.parent / "results" / "benchmark_results.json"
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {output_path}")
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

