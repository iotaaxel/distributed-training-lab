"""Simple ResNet implementation for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convs."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    Small ResNet for CIFAR-10.
    
    Configurable depth via num_layers parameter.
    Kept small to run quickly on 1-4 GPUs without huge memory requirements.
    """
    
    def __init__(self, num_layers: int = 18, num_classes: int = 10):
        """
        Args:
            num_layers: Total number of layers (18, 34, etc.)
            num_classes: Number of output classes (10 for CIFAR-10)
        """
        super().__init__()
        self.in_planes = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        # For CIFAR-10, we use smaller strides than ImageNet ResNet
        self.layer1 = self._make_layer(BasicBlock, 64, num_layers // 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_layers // 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_layers // 3, stride=2)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)
        
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        """Create a layer with num_blocks blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_model(model_name: str = "small_resnet", num_layers: int = 18, num_classes: int = 10) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of model to create
        num_layers: Number of layers (for ResNet)
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if model_name == "small_resnet":
        return SmallResNet(num_layers=num_layers, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

