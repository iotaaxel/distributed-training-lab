"""Basic tests for trainers."""

import torch
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimpleModel
from src.training import DDPTrainer, FSDPTrainer
from src.utils.config import get_default_config


@pytest.fixture
def config():
    """Get default config for testing."""
    config = get_default_config()
    # Use smaller model for faster tests
    config["model"]["hidden_size"] = 256
    config["model"]["num_layers"] = 2
    config["benchmark"]["warmup_steps"] = 2
    config["benchmark"]["profile_steps"] = 5
    return config


@pytest.fixture
def device():
    """Get device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_model_forward(config, device):
    """Test that model can do forward pass."""
    model = SimpleModel(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    ).to(device)
    
    batch_size = config["training"]["batch_size"]
    seq_len = config["training"]["seq_len"]
    vocab_size = config["model"]["vocab_size"]
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits = model(x)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_ddp_trainer_single_gpu(config, device):
    """Test DDP trainer on single GPU."""
    if device.type == 'cpu':
        pytest.skip("Skipping GPU test on CPU")
    
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
        rank=0,
        world_size=1,
    )
    
    # Run a few steps
    summary = trainer.train(num_steps=5, warmup_steps=2)
    
    assert "avg_step_time" in summary
    assert "avg_loss" in summary


def test_fsdp_trainer_single_gpu(config, device):
    """Test FSDP trainer on single GPU."""
    if device.type == 'cpu':
        pytest.skip("Skipping GPU test on CPU")
    
    model = SimpleModel(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
    )
    
    trainer = FSDPTrainer(
        model=model,
        config=config,
        device=device,
        rank=0,
        world_size=1,
    )
    
    # Run a few steps
    summary = trainer.train(num_steps=5, warmup_steps=2)
    
    assert "avg_step_time" in summary
    assert "avg_loss" in summary

