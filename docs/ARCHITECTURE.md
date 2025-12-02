# Architecture Documentation

## Overview

This document describes the architecture and design decisions of the Distributed Training Lab.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Scripts                        │
│  (train_ddp.py, train_fsdp.py, benchmark.py)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Trainer Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ BaseTrainer  │  │  DDPTrainer  │  │  FSDPTrainer │     │
│  │              │  │              │  │              │     │
│  │ - train_step │  │ - DDP wrap   │  │ - FSDP wrap   │     │
│  │ - checkpoint │  │ - allreduce  │  │ - allgather   │     │
│  │ - scheduler  │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│  ┌────────────────────────────────────────────────────┐   │
│  │              SimpleModel (Transformer)              │   │
│  │  - Embedding                                         │   │
│  │  - N × TransformerBlock                              │   │
│  │  - Output Head                                       │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Benchmarking & Metrics                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Metrics    │  │  Profiler    │  │ Communication│     │
│  │  Collector   │  │              │  │   Profiler   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Modularity

Each component is self-contained and can be used independently:
- **BaseTrainer**: Common training logic (loss computation, checkpointing, scheduling)
- **DDPTrainer**: DDP-specific wrapping and gradient synchronization
- **FSDPTrainer**: FSDP-specific sharding and communication

### 2. Extensibility

The architecture makes it easy to:
- Add new distributed strategies (e.g., DeepSpeed, ZeRO)
- Swap in different models
- Add new metrics or profiling tools

### 3. Production Readiness

- Comprehensive error handling
- Configuration validation
- Checkpointing and resume capability
- Proper logging and metrics collection

## Key Components

### BaseTrainer

The foundation for all trainers, providing:
- Training loop with gradient accumulation
- Learning rate scheduling
- Checkpointing (save/load)
- Metrics collection integration
- Synthetic data generation

### DDP Implementation

**How it works:**
1. Model is replicated on each GPU
2. Each GPU processes different data batches
3. After backward pass, gradients are allreduced across GPUs
4. Each GPU updates its model copy with averaged gradients

**Communication Pattern:**
```
Step 1: Forward pass (parallel, no communication)
Step 2: Backward pass (parallel, no communication)
Step 3: Allreduce gradients (synchronization point)
Step 4: Optimizer step (parallel, no communication)
```

**Memory Usage:**
- Each GPU holds: Full model + Full gradients + Full optimizer states
- Total memory: `N_GPUs × (Model + Gradients + Optimizer)`

### FSDP Implementation

**How it works:**
1. Model parameters are sharded across GPUs
2. During forward: Allgather sharded parameters → compute → discard
3. During backward: Allgather parameters → compute gradients → Allreduce gradients → discard
4. Optimizer states are also sharded

**Communication Pattern:**
```
Step 1: Allgather parameters (synchronization)
Step 2: Forward pass (parallel)
Step 3: Discard gathered parameters
Step 4: Allgather parameters for backward (synchronization)
Step 5: Backward pass (parallel)
Step 6: Allreduce gradients (synchronization)
Step 7: Discard gathered parameters
```

**Memory Usage:**
- Each GPU holds: `1/N_GPUs` of model + `1/N_GPUs` of gradients + `1/N_GPUs` of optimizer states
- Plus temporary gathered parameters during forward/backward
- Total memory: `(Model + Gradients + Optimizer) / N_GPUs + temporary buffers`

## Performance Characteristics

### DDP

**Strengths:**
- Low communication overhead (one allreduce per step)
- Simple implementation
- Best throughput for models that fit on GPU

**Weaknesses:**
- Cannot scale beyond single GPU memory
- Memory inefficient (full replication)

### FSDP

**Strengths:**
- Can train models larger than single GPU
- Memory efficient (sharding)
- Scales to very large models

**Weaknesses:**
- Higher communication overhead (multiple allgathers)
- More complex implementation
- May have lower throughput for small models

## Communication Analysis

### DDP Communication Cost

Per training step:
- **1 allreduce** operation
- Size: `num_parameters × 4 bytes` (float32)
- Bandwidth: Depends on interconnect (NVLink, InfiniBand, etc.)

### FSDP Communication Cost

Per training step:
- **2 allgather** operations (forward + backward)
- **1 allreduce** operation (gradients)
- Size varies based on sharding strategy

**FULL_SHARD:**
- Forward: `(num_parameters / N_GPUs) × N_GPUs × 4 bytes`
- Backward: Same as forward
- Gradient sync: `num_parameters × 4 bytes`

## Memory Analysis

### Model Memory Breakdown

For a transformer model:
- **Parameters**: `12 × hidden_size²` (roughly, depends on architecture)
- **Gradients**: Same size as parameters
- **Optimizer states** (AdamW): `2 × parameters` (momentum + variance)
- **Activations**: `batch_size × seq_len × hidden_size × num_layers`

### DDP Memory

```
Per GPU = Parameters + Gradients + Optimizer States + Activations
Total = N_GPUs × Per GPU
```

### FSDP Memory

```
Per GPU = (Parameters + Gradients + Optimizer States) / N_GPUs + Activations + Temporary Buffers
Total ≈ Parameters + Gradients + Optimizer States + N_GPUs × Activations
```

## Best Practices

1. **Model Size vs GPU Memory:**
   - If model fits on 1 GPU → Use DDP
   - If model exceeds 1 GPU → Use FSDP

2. **Batch Size:**
   - Larger batch sizes improve GPU utilization
   - Balance between memory and convergence

3. **Gradient Accumulation:**
   - Use when effective batch size needs to be larger
   - Reduces communication frequency

4. **Mixed Precision:**
   - Can reduce memory usage by ~50%
   - May improve throughput
   - Use with FSDP for very large models

5. **Checkpointing:**
   - Save checkpoints regularly
   - FSDP requires special handling to gather sharded parameters

## Future Enhancements

Potential additions:
- **DeepSpeed ZeRO** integration
- **Pipeline Parallelism** support
- **Tensor Parallelism** support
- **Hybrid parallelism** (data + model + pipeline)
- **Automatic mixed precision** with loss scaling
- **Gradient checkpointing** for memory efficiency
- **Communication overlap** optimizations

