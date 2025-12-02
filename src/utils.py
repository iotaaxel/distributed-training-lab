"""Utility functions for logging, timing, and seeding."""

import time
import random
import numpy as np
import torch
import logging
import sys
from typing import Optional


def setup_logging(rank: int = 0, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Only rank 0 logs to avoid duplicate output in distributed training.
    This is a common pattern - you want one clean log stream, not N copies.
    
    Args:
        rank: Process rank (0 for single GPU)
        level: Logging level
    """
    if rank != 0:
        # Suppress logging from non-zero ranks
        logging.basicConfig(level=logging.WARNING)
        return
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Important for benchmarking - ensures consistent results across runs.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """Simple timer context manager for measuring code execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start_time
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed if self.elapsed is not None else 0.0


def get_peak_memory_mb(device: torch.device) -> float:
    """
    Get peak GPU memory usage in MB.
    
    Uses torch.cuda.max_memory_allocated() which tracks the maximum
    memory allocated by tensors. This is useful for comparing memory
    efficiency between DDP and FSDP.
    
    Args:
        device: PyTorch device
        
    Returns:
        Peak memory in MB, or 0.0 if not CUDA device
    """
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    return 0.0


def reset_peak_memory(device: torch.device):
    """Reset peak memory tracking."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

