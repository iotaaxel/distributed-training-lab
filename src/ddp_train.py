"""DDP (DistributedDataParallel) training script for CIFAR-10."""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from pathlib import Path

from data import get_cifar10_dataset, get_dataloader
from models import create_model
from utils import setup_logging, set_seed, Timer, get_peak_memory_mb, reset_peak_memory


def setup_distributed():
    """
    Initialize distributed training environment.
    
    Uses torchrun (or manual env vars) to set up process group.
    For single GPU, world_size=1 and this becomes a no-op.
    """
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
        # Initialize process group
        # NCCL backend is fastest for multi-GPU on same node
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


def train_epoch(model, train_loader, criterion, optimizer, device, rank, log_interval=100):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        rank: Process rank (for logging)
        log_interval: Steps between logging
        
    Returns:
        Average loss and samples processed
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    step_times = []
    
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        step_start = time.time()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        # DDP automatically synchronizes gradients here via allreduce
        # This is the key difference from regular training - gradients are
        # averaged across all GPUs before optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        
        # Accumulate metrics
        total_loss += loss.item()
        batch_size = images.size(0)
        num_samples += batch_size * (dist.get_world_size() if dist.is_initialized() else 1)
        step_times.append(step_time)
        
        # Logging (only rank 0 to avoid duplicate output)
        if rank == 0 and (step + 1) % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            avg_step_time = sum(step_times) / len(step_times)
            samples_per_sec = batch_size / avg_step_time
            print(f"Step {step + 1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Step time: {avg_step_time*1000:.2f}ms | "
                  f"Samples/sec: {samples_per_sec:.0f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, num_samples, step_times


def main():
    parser = argparse.ArgumentParser(description="DDP training on CIFAR-10")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    try:
        # Setup logging and seeding
        setup_logging(rank=rank)
        set_seed(42)
        
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if rank == 0:
            print("="*60)
            print("DDP Training")
            print("="*60)
            print(f"World size: {world_size}")
            print(f"Device: {device}")
            print(f"Config: {config}")
            print()
        
        # Create model
        model = create_model(
            model_name=config["model"]["name"],
            num_layers=config["model"].get("num_layers", 18),
            num_classes=10
        )
        model = model.to(device)
        
        # Wrap with DDP
        # DDP replicates the full model on each GPU and synchronizes gradients
        # This is the key: each GPU has a complete copy of model params
        if world_size > 1:
            model = DDP(model, device_ids=[device.index], output_device=device.index)
            if rank == 0:
                print(f"DDP initialized: {world_size} GPUs")
        else:
            if rank == 0:
                print("Single GPU mode (DDP not needed)")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=0.9,
            weight_decay=5e-4
        ) if config["training"].get("optimizer", "sgd").lower() == "sgd" else optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )
        
        # Data
        train_dataset = get_cifar10_dataset(root="./data", train=True)
        train_loader = get_dataloader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            rank=rank,
            world_size=world_size,
        )
        
        # Training loop
        num_epochs = config["training"]["num_epochs"]
        all_step_times = []
        total_samples = 0
        
        reset_peak_memory(device)
        
        if rank == 0:
            print(f"\nTraining for {num_epochs} epochs...")
        
        with Timer("Training"):
            for epoch in range(num_epochs):
                if world_size > 1:
                    # Set epoch for DistributedSampler (ensures different shuffle each epoch)
                    train_loader.sampler.set_epoch(epoch)
                
                avg_loss, samples, step_times = train_epoch(
                    model, train_loader, criterion, optimizer, device, rank
                )
                all_step_times.extend(step_times)
                total_samples += samples
                
                if rank == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
        
        # Summary statistics
        if rank == 0:
            peak_memory_mb = get_peak_memory_mb(device)
            avg_step_time = sum(all_step_times) / len(all_step_times) if all_step_times else 0.0
            total_time = sum(all_step_times)
            images_per_sec = total_samples / total_time if total_time > 0 else 0.0
            
            print("\n" + "="*60)
            print("Training Summary")
            print("="*60)
            print(f"Total samples processed: {total_samples}")
            print(f"Average step time: {avg_step_time:.4f}s")
            print(f"Images/sec: {images_per_sec:.0f}")
            print(f"Peak GPU memory: {peak_memory_mb:.0f} MB")
            print("="*60)
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

