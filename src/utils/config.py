"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "model": {
            "vocab_size": 10000,
            "hidden_size": 1024,
            "num_layers": 12,
            "num_heads": 16,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 32,
            "seq_len": 512,
            "num_epochs": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
        },
        "data": {
            "synthetic": True,
        },
        "benchmark": {
            "warmup_steps": 10,
            "profile_steps": 50,
        },
    }

