"""Metrics collection for benchmarking."""

import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import psutil
import os


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    step_time: float  # seconds
    loss: float
    memory_allocated: float  # GB
    memory_reserved: float  # GB
    cpu_percent: float
    gpu_utilization: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "step_time": self.step_time,
            "loss": self.loss,
            "memory_allocated_gb": self.memory_allocated,
            "memory_reserved_gb": self.memory_reserved,
            "cpu_percent": self.cpu_percent,
            "gpu_utilization": self.gpu_utilization,
        }


class MetricsCollector:
    """Collects and aggregates training metrics."""
    
    def __init__(self):
        self.metrics: List[StepMetrics] = []
        self.process = psutil.Process(os.getpid())
        
    def record_step(
        self,
        step: int,
        step_time: float,
        loss: float,
        device: torch.device,
    ) -> StepMetrics:
        """
        Record metrics for a single step.
        
        Args:
            step: Step number
            step_time: Time taken for the step (seconds)
            loss: Loss value
            device: PyTorch device
            
        Returns:
            StepMetrics object
        """
        # Memory metrics
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9  # GB
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        
        metric = StepMetrics(
            step=step,
            step_time=step_time,
            loss=loss,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            cpu_percent=cpu_percent,
        )
        
        self.metrics.append(metric)
        return metric
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of collected metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics:
            return {}
        
        step_times = [m.step_time for m in self.metrics]
        losses = [m.loss for m in self.metrics]
        memory_allocated = [m.memory_allocated for m in self.metrics]
        memory_reserved = [m.memory_reserved for m in self.metrics]
        
        return {
            "num_steps": len(self.metrics),
            "avg_step_time": sum(step_times) / len(step_times),
            "min_step_time": min(step_times),
            "max_step_time": max(step_times),
            "throughput_samples_per_sec": 1.0 / (sum(step_times) / len(step_times)),
            "avg_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "peak_memory_allocated_gb": max(memory_allocated),
            "avg_memory_allocated_gb": sum(memory_allocated) / len(memory_allocated),
            "peak_memory_reserved_gb": max(memory_reserved),
            "avg_memory_reserved_gb": sum(memory_reserved) / len(memory_reserved),
        }
    
    def reset(self) -> None:
        """Reset collected metrics."""
        self.metrics.clear()

