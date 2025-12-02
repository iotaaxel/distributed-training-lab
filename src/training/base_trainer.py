"""Base trainer class with common functionality."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time
import logging

from ..benchmark import MetricsCollector, Profiler
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Base class for distributed trainers.
    
    Provides common functionality for training loops, metrics collection,
    and checkpointing. Subclasses implement distributed-specific setup.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize base trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration dictionary
            device: Device to train on
            rank: Process rank in distributed setup (0 for single GPU)
            world_size: Total number of processes
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        
        # Setup logging
        setup_logging(rank=rank)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"].get("weight_decay", 0.01),
            betas=config["training"].get("betas", (0.9, 0.999)),
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint", {}).get("dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on config."""
        scheduler_config = self.config["training"].get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "none")
        
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", 1000),
                eta_min=scheduler_config.get("eta_min", 0.0),
            )
        elif scheduler_type == "linear_warmup":
            from torch.optim.lr_scheduler import LambdaLR
            
            warmup_steps = scheduler_config.get("warmup_steps", 100)
            total_steps = scheduler_config.get("total_steps", 10000)
            
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
                )
            
            return LambdaLR(self.optimizer, lr_lambda)
        else:
            return None
    
    @abstractmethod
    def setup_distributed(self) -> None:
        """Setup distributed training (implemented by subclasses)."""
        pass
    
    def generate_synthetic_batch(self) -> torch.Tensor:
        """
        Generate a synthetic batch of data for benchmarking.
        
        Returns:
            Batch of token IDs with shape [batch_size, seq_len]
        """
        batch_size = self.config["training"]["batch_size"]
        seq_len = self.config["training"]["seq_len"]
        vocab_size = self.config["model"]["vocab_size"]
        
        # Generate random token IDs
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        return data
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for next-token prediction.
        
        Args:
            logits: Model output with shape [batch_size, seq_len, vocab_size]
            targets: Target token IDs with shape [batch_size, seq_len]
            
        Returns:
            Scalar loss tensor
        """
        # Shift targets for next-token prediction
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,  # Allow for padding tokens if needed
        )
        return loss
    
    def train_step(self) -> Dict[str, float]:
        """
        Execute a single training step with gradient accumulation support.
        
        Returns:
            Dictionary with step metrics (loss, learning_rate, etc.)
        """
        accumulated_loss = 0.0
        
        for accumulation_step in range(self.gradient_accumulation_steps):
            # Generate synthetic batch
            data = self.generate_synthetic_batch()
            
            # Forward pass
            logits = self.model(data)
            loss = self.compute_loss(logits, data)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
        
        # Gradient clipping (optional)
        max_grad_norm = self.config["training"].get("max_grad_norm", None)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        
        avg_loss = accumulated_loss / self.gradient_accumulation_steps
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        return {
            "loss": avg_loss,
            "learning_rate": current_lr,
        }
    
    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Save training checkpoint.
        
        Args:
            path: Optional path to save checkpoint. If None, uses default naming.
            
        Returns:
            Path to saved checkpoint
        """
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self._get_model_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "metrics": self.metrics_collector.get_summary(),
        }
        
        torch.save(checkpoint, path)
        
        if self.rank == 0:
            logger.info(f"Checkpoint saved to {path}")
        
        return path
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self._load_model_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        
        if self.rank == 0:
            logger.info(f"Checkpoint loaded from {path} (step {self.global_step})")
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get model state dict (handles DDP/FSDP wrapping)."""
        if hasattr(self.model, "module"):  # DDP wrapped
            return self.model.module.state_dict()
        elif hasattr(self.model, "_fsdp_wrapped_module"):  # FSDP wrapped
            # For FSDP, we need to gather sharded parameters
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if isinstance(self.model, FSDP):
                with FSDP.summon_full_params(self.model):
                    return self.model.state_dict()
        return self.model.state_dict()
    
    def _load_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load model state dict (handles DDP/FSDP wrapping)."""
        if hasattr(self.model, "module"):  # DDP wrapped
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
    
    def train(
        self,
        num_steps: Optional[int] = None,
        warmup_steps: int = 10,
        profile: bool = False,
        save_checkpoints: bool = False,
        checkpoint_interval: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run training loop with benchmarking.
        
        Args:
            num_steps: Number of training steps (defaults to config)
            warmup_steps: Number of warmup steps before benchmarking
            profile: Whether to enable PyTorch profiling
            save_checkpoints: Whether to save checkpoints during training
            checkpoint_interval: Steps between checkpoints
            
        Returns:
            Dictionary with training summary metrics
        """
        if num_steps is None:
            num_steps = self.config["benchmark"]["profile_steps"]
        
        total_steps = warmup_steps + num_steps
        
        # Warmup phase
        if self.rank == 0:
            logger.info(f"Warming up for {warmup_steps} steps...")
        
        for step in range(warmup_steps):
            self.train_step()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmarking phase
        if self.rank == 0:
            logger.info(f"Benchmarking for {num_steps} steps...")
        
        with Profiler(enabled=profile, output_dir="./profiles"):
            for step in range(warmup_steps, total_steps):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                step_start = time.time()
                
                step_metrics = self.train_step()
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                step_time = time.time() - step_start
                
                # Record metrics
                self.metrics_collector.record_step(
                    step=self.global_step,
                    step_time=step_time,
                    loss=step_metrics["loss"],
                    device=self.device,
                )
                
                # Periodic logging
                if self.rank == 0 and (step + 1) % 10 == 0:
                    logger.info(
                        f"Step {step + 1}/{total_steps} | "
                        f"Loss: {step_metrics['loss']:.4f} | "
                        f"LR: {step_metrics['learning_rate']:.2e} | "
                        f"Time: {step_time*1000:.2f}ms"
                    )
                
                # Save checkpoint
                if save_checkpoints and (step + 1) % checkpoint_interval == 0:
                    self.save_checkpoint()
        
        # Get summary
        summary = self.metrics_collector.get_summary()
        summary["total_steps"] = self.global_step
        summary["final_learning_rate"] = self.optimizer.param_groups[0]["lr"]
        
        if self.rank == 0:
            logger.info("\n" + "="*50)
            logger.info("Training Summary")
            logger.info("="*50)
            for key, value in summary.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
        
        return summary
