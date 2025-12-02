"""A simple model for benchmarking distributed training strategies."""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """
    A configurable model for benchmarking DDP vs FSDP.
    
    This model uses a stack of transformer-like blocks to create
    a model that can scale in size for memory/performance testing.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights (common in language models)
        self.head.weight = self.embedding.weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        return logits


class TransformerBlock(nn.Module):
    """A single transformer block with attention and MLP."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)
        
        return x

