"""Configuration validation utilities."""

from typing import Dict, Any, List, Tuple
import torch


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate training configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required sections
    required_sections = ["model", "training", "benchmark"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate model config
    if "model" in config:
        model_config = config["model"]
        if "hidden_size" in model_config:
            if model_config["hidden_size"] <= 0:
                errors.append("model.hidden_size must be positive")
        
        if "num_layers" in model_config:
            if model_config["num_layers"] <= 0:
                errors.append("model.num_layers must be positive")
        
        if "num_heads" in model_config:
            if model_config["num_heads"] <= 0:
                errors.append("model.num_heads must be positive")
            if "hidden_size" in model_config:
                if model_config["hidden_size"] % model_config["num_heads"] != 0:
                    errors.append(
                        f"model.hidden_size ({model_config['hidden_size']}) "
                        f"must be divisible by model.num_heads ({model_config['num_heads']})"
                    )
    
    # Validate training config
    if "training" in config:
        training_config = config["training"]
        if "batch_size" in training_config:
            if training_config["batch_size"] <= 0:
                errors.append("training.batch_size must be positive")
        
        if "learning_rate" in training_config:
            if training_config["learning_rate"] <= 0:
                errors.append("training.learning_rate must be positive")
    
    # Validate benchmark config
    if "benchmark" in config:
        benchmark_config = config["benchmark"]
        if "warmup_steps" in benchmark_config:
            if benchmark_config["warmup_steps"] < 0:
                errors.append("benchmark.warmup_steps must be non-negative")
        
        if "profile_steps" in benchmark_config:
            if benchmark_config["profile_steps"] <= 0:
                errors.append("benchmark.profile_steps must be positive")
    
    # Validate FSDP config if present
    if "fsdp" in config:
        fsdp_config = config["fsdp"]
        if "sharding_strategy" in fsdp_config:
            valid_strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
            if fsdp_config["sharding_strategy"] not in valid_strategies:
                errors.append(
                    f"fsdp.sharding_strategy must be one of {valid_strategies}, "
                    f"got {fsdp_config['sharding_strategy']}"
                )
    
    return len(errors) == 0, errors


def check_gpu_availability(num_gpus: int) -> Tuple[bool, str]:
    """
    Check if requested number of GPUs are available.
    
    Args:
        num_gpus: Number of GPUs requested
        
    Returns:
        Tuple of (is_available, error_message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        return False, (
            f"Requested {num_gpus} GPUs but only {available_gpus} available"
        )
    
    return True, ""

