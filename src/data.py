"""Dataset and dataloader utilities for CIFAR-10."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torch.distributed as dist


def get_cifar10_dataset(root: str = "./data", train: bool = True):
    """
    Load CIFAR-10 dataset.
    
    Args:
        root: Root directory for dataset
        train: Whether to load training or test set
        
    Returns:
        CIFAR-10 dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Create a DataLoader, optionally with DistributedSampler.
    
    DistributedSampler ensures each GPU gets a different subset of data.
    This is critical for correct distributed training - without it, all GPUs
    would see the same data, defeating the purpose of data parallelism.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle (ignored if world_size > 1, uses sampler instead)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        rank: Process rank (for DistributedSampler)
        world_size: Number of processes (for DistributedSampler)
        
    Returns:
        DataLoader
    """
    if world_size > 1:
        # DistributedSampler handles data partitioning across GPUs
        # Each rank gets a different subset: rank 0 gets samples [0, world_size, 2*world_size, ...]
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Ensures consistent batch sizes across GPUs
    )
    
    return dataloader

