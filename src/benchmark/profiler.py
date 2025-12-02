"""Profiling utilities for performance analysis."""

import torch
from typing import Optional, ContextManager


class Profiler:
    """Simple profiler wrapper for PyTorch profiling."""
    
    def __init__(self, enabled: bool = False, output_dir: Optional[str] = None):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
            output_dir: Directory to save profiling traces
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.profiler: Optional[torch.profiler.profile] = None
        
    def __enter__(self) -> ContextManager:
        """Context manager entry."""
        if self.enabled:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.output_dir
                ) if self.output_dir else None,
                record_shapes=True,
                with_stack=True,
            )
            self.profiler.__enter__()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        if self.profiler is not None:
            self.profiler.__exit__(*args)
    
    def step(self) -> None:
        """Advance profiler step."""
        if self.profiler is not None:
            self.profiler.step()

