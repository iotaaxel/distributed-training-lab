"""Basic tests for training scripts."""

import torch
import pytest
from pathlib import Path
import sys
import os

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import from the flat module files
from models import create_model
from data import get_cifar10_dataset, get_dataloader


@pytest.fixture
def device():
    """Get device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_model_forward(device):
    """Test that model can do forward pass."""
    model = create_model(
        model_name="small_resnet",
        num_layers=18,
        num_classes=10
    ).to(device)
    
    # Create dummy input (CIFAR-10: 3 channels, 32x32)
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    logits = model(x)
    
    assert logits.shape == (batch_size, 10)  # 10 classes for CIFAR-10


def test_dataloader_creation():
    """Test that dataloader can be created."""
    dataset = get_cifar10_dataset(root="./data", train=True)
    dataloader = get_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        rank=0,
        world_size=1,
    )
    
    # Check that we can get a batch
    batch = next(iter(dataloader))
    images, labels = batch
    assert images.shape[0] == 32  # batch size
    assert images.shape[1] == 3   # RGB channels
    assert images.shape[2] == 32  # height
    assert images.shape[3] == 32  # width
    assert labels.shape[0] == 32  # batch size
