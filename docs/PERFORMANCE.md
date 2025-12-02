# Performance Analysis Guide

## Understanding Benchmark Results

### Key Metrics

1. **Step Time**: Time per training step (lower is better)
   - Includes forward, backward, and communication
   - Measured in milliseconds

2. **Throughput**: Samples processed per second (higher is better)
   - `throughput = batch_size × world_size / step_time`
   - Measures training efficiency

3. **Memory Usage**: Peak GPU memory (lower is better for larger models)
   - **Allocated**: Memory actively used by tensors
   - **Reserved**: Memory reserved by PyTorch (may be larger)

4. **Communication Overhead**: Time spent in distributed operations
   - DDP: Allreduce time
   - FSDP: Allgather + Allreduce time

## Expected Performance Characteristics

### Small Models (< 1B parameters, fits on 1 GPU)

**DDP:**
- ✅ Faster step time (lower communication overhead)
- ✅ Higher throughput
- ❌ Full model replication (memory inefficient)

**FSDP:**
- ❌ Slower step time (higher communication overhead)
- ❌ Lower throughput
- ✅ More memory efficient (but not needed)

**Recommendation:** Use DDP

### Large Models (> 1B parameters, exceeds 1 GPU)

**DDP:**
- ❌ Cannot fit model (OOM error)

**FSDP:**
- ✅ Can fit model (sharding)
- ✅ Memory efficient
- ⚠️ Higher communication overhead (acceptable tradeoff)

**Recommendation:** Use FSDP

### Medium Models (fits on 1 GPU, but tight)

**DDP:**
- ✅ Faster if it fits
- ❌ May have memory pressure

**FSDP:**
- ✅ More headroom for larger batches
- ⚠️ Slightly slower

**Recommendation:** Test both, choose based on batch size needs

## Performance Optimization Tips

### 1. Batch Size Tuning

- **Larger batches** → Better GPU utilization
- **Too large** → OOM errors
- **Too small** → Underutilized GPUs

**Strategy:**
1. Start with batch_size=32
2. Double until OOM
3. Use gradient accumulation to simulate larger batches

### 2. Gradient Accumulation

When effective batch size needs to be larger:

```python
# Instead of batch_size=128
batch_size=32
gradient_accumulation_steps=4
# Effective batch size = 32 × 4 = 128
```

**Benefits:**
- Same effective batch size
- Lower memory usage
- Fewer communication operations (if accumulation > 1)

### 3. Mixed Precision Training

Enable in FSDP config:

```yaml
fsdp:
  use_mixed_precision: true
```

**Benefits:**
- ~50% memory reduction
- Often 1.5-2x speedup
- Minimal accuracy impact

### 4. Learning Rate Scheduling

Warmup + decay improves convergence:

```yaml
training:
  scheduler:
    type: "linear_warmup"
    warmup_steps: 1000
    total_steps: 10000
```

### 5. Communication Optimization

**For DDP:**
- Use `find_unused_parameters=False` if possible (faster)
- Ensure good interconnect (NVLink, InfiniBand)

**For FSDP:**
- Use `FULL_SHARD` for maximum memory savings
- Consider `SHARD_GRAD_OP` for lower communication overhead
- Enable `use_orig_params=True` for better performance (PyTorch 2.1+)

## Benchmarking Methodology

### 1. Warmup Phase

Always include warmup steps to:
- Initialize CUDA kernels
- Warm up communication primitives
- Stabilize memory allocation

**Recommendation:** 10-50 warmup steps

### 2. Measurement Phase

Measure enough steps to get stable averages:

**Too few steps:**
- High variance
- Unreliable results

**Too many steps:**
- Wastes time
- May hit memory leaks

**Recommendation:** 50-200 steps for benchmarking

### 3. Multiple Runs

Run benchmarks multiple times and average:
- First run may be slower (JIT compilation, etc.)
- System state varies
- Network conditions vary

**Recommendation:** 3-5 runs, report mean ± std

## Interpreting Results

### Example Output

```
Training Summary
==================================================
num_steps: 50
avg_step_time: 0.0452
min_step_time: 0.0431
max_step_time: 0.0489
throughput_samples_per_sec: 22.12
avg_loss: 8.2341
final_loss: 7.8923
peak_memory_allocated_gb: 12.45
avg_memory_allocated_gb: 12.32
```

### Analysis

1. **Step Time Variance:**
   - `max - min = 0.0058s` (13% variance)
   - High variance may indicate system interference

2. **Throughput:**
   - `22.12 samples/sec` with batch_size=32, world_size=4
   - Effective: `22.12 × 32 × 4 = 2,831 tokens/sec`

3. **Memory:**
   - Peak: 12.45 GB
   - Average: 12.32 GB
   - Stable (good, no leaks)

## Common Performance Issues

### 1. Low GPU Utilization

**Symptoms:**
- Step time much higher than expected
- GPU utilization < 80%

**Causes:**
- Batch size too small
- Data loading bottleneck
- CPU-GPU transfer overhead

**Solutions:**
- Increase batch size
- Use `pin_memory=True` for data loaders
- Profile with `torch.profiler`

### 2. High Communication Overhead

**Symptoms:**
- FSDP much slower than DDP for small models
- Step time increases with world_size

**Solutions:**
- Use DDP for models that fit
- Consider `SHARD_GRAD_OP` instead of `FULL_SHARD`
- Ensure good interconnect (NVLink > PCIe)

### 3. Memory Fragmentation

**Symptoms:**
- OOM errors even when total memory seems available
- Memory usage increases over time

**Solutions:**
- Use gradient checkpointing
- Reduce batch size
- Enable mixed precision
- Restart training periodically

## Profiling

### PyTorch Profiler

Enable profiling in training script:

```python
trainer.train(profile=True)
```

View results:
```bash
tensorboard --logdir=./profiles
```

### Communication Profiling

Use `CommunicationProfiler` to analyze communication costs:

```python
from src.benchmark.communication import CommunicationProfiler

profiler = CommunicationProfiler(enabled=True)
ddp_cost = profiler.estimate_ddp_comm_cost(model, world_size=4)
fsdp_cost = profiler.estimate_fsdp_comm_cost(model, world_size=4)
```

## Scaling Analysis

### Strong Scaling (Fixed Problem Size)

As you add GPUs:
- **DDP**: Throughput should scale linearly (up to communication limits)
- **FSDP**: Throughput may not scale as well (higher comm overhead)

### Weak Scaling (Fixed Problem Size per GPU)

As you add GPUs:
- **DDP**: Problem size per GPU stays same, total problem size increases
- **FSDP**: Can increase total problem size (larger model)

## References

- [PyTorch DDP Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [FSDP Performance Best Practices](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Communication-Efficient Distributed Training](https://arxiv.org/abs/1710.06952)

