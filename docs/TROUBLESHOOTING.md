# Troubleshooting Guide

## Common Issues and Solutions

### 1. CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB.
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 16  # Reduce from 32
   ```

2. **Use gradient accumulation:**
   ```python
   trainer = DDPTrainer(
       model=model,
       config=config,
       device=device,
       rank=rank,
       world_size=world_size,
       gradient_accumulation_steps=2,  # Accumulate over 2 steps
   )
   ```

3. **Reduce model size:**
   ```yaml
   model:
     hidden_size: 512  # Reduce from 1024
     num_layers: 6     # Reduce from 12
   ```

4. **Use FSDP instead of DDP:**
   - FSDP shards parameters, allowing larger models

5. **Enable mixed precision (FSDP):**
   ```yaml
   fsdp:
     use_mixed_precision: true
   ```

6. **Clear cache between runs:**
   ```python
   torch.cuda.empty_cache()
   ```

### 2. NCCL Errors

**Symptoms:**
```
RuntimeError: NCCL error: unhandled system error
NCCL error: initialization failed
```

**Solutions:**

1. **Check GPU visibility:**
   ```bash
   nvidia-smi
   ```

2. **Ensure all GPUs are accessible:**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

3. **Set NCCL environment variables:**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1  # If InfiniBand issues
   ```

4. **Use TCP backend (if needed):**
   ```python
   dist.init_process_group(backend="gloo")  # Slower but more compatible
   ```

5. **Check firewall/network:**
   - Ensure GPUs can communicate
   - Check for port conflicts

### 3. Hanging/Deadlock

**Symptoms:**
- Training starts but hangs
- No progress after initialization

**Solutions:**

1. **Check all processes are synchronized:**
   - Ensure all ranks reach the same point
   - Add logging to identify where it hangs

2. **Verify data loading:**
   - Synthetic data should be fast
   - Real data loaders may cause hangs if not properly distributed

3. **Check for uneven work:**
   - All ranks should process same number of steps
   - Avoid rank-specific early exits

4. **Use timeout:**
   ```python
   dist.init_process_group(
       backend="nccl",
       timeout=timedelta(seconds=1800),  # 30 min timeout
   )
   ```

### 4. Slow Performance

**Symptoms:**
- Step time much higher than expected
- Low GPU utilization

**Solutions:**

1. **Check batch size:**
   - Too small → underutilized GPUs
   - Too large → memory issues

2. **Profile with torch.profiler:**
   ```python
   trainer.train(profile=True)
   # View in tensorboard
   ```

3. **Check communication overhead:**
   - DDP should have minimal overhead
   - FSDP has higher overhead (expected)

4. **Verify GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Check for CPU bottlenecks:**
   - Monitor CPU usage
   - Synthetic data should be fast

### 5. Loss Not Decreasing

**Symptoms:**
- Loss stays constant or increases
- Model not learning

**Solutions:**

1. **Check learning rate:**
   ```yaml
   training:
     learning_rate: 1e-4  # Try different values
   ```

2. **Verify gradient flow:**
   ```python
   # Add gradient norm logging
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   print(f"Gradient norm: {total_norm}")
   ```

3. **Check loss computation:**
   - Verify targets are correct
   - Check for NaN/Inf values

4. **Use learning rate scheduling:**
   ```yaml
   training:
     scheduler:
       type: "linear_warmup"
       warmup_steps: 1000
   ```

### 6. Checkpoint Loading Issues

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
KeyError: 'module.layer.weight'
```

**Solutions:**

1. **DDP checkpoint loading:**
   - If saved with DDP, load with DDP
   - If saved without DDP, may need to add 'module.' prefix

2. **FSDP checkpoint loading:**
   - FSDP requires special handling
   - Use `FSDP.summon_full_params()` when saving

3. **Check device mapping:**
   ```python
   checkpoint = torch.load(path, map_location=device)
   ```

### 7. Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
```

**Solutions:**

1. **Run from repository root:**
   ```bash
   cd /path/to/distributed-training-lab
   python scripts/train_ddp.py
   ```

2. **Add to PYTHONPATH:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/distributed-training-lab"
   ```

3. **Install in development mode:**
   ```bash
   pip install -e .
   ```

### 8. Configuration Validation Errors

**Symptoms:**
```
Configuration errors:
  - model.hidden_size must be divisible by model.num_heads
```

**Solutions:**

1. **Fix configuration:**
   ```yaml
   model:
     hidden_size: 1024  # Must be divisible by num_heads
     num_heads: 16       # 1024 / 16 = 64 ✓
   ```

2. **Use validation:**
   ```python
   from src.utils.validation import validate_config
   is_valid, errors = validate_config(config)
   ```

### 9. Mixed Precision Issues

**Symptoms:**
- NaN losses
- Training instability

**Solutions:**

1. **Disable mixed precision:**
   ```yaml
   fsdp:
     use_mixed_precision: false
   ```

2. **Use gradient scaling:**
   - FSDP handles this automatically
   - For manual implementation, use `GradScaler`

3. **Check for operations not supporting FP16:**
   - Some operations require FP32
   - FSDP handles this with `buffer_dtype=torch.float32`

### 10. Multi-Node Issues

**Note:** This repo focuses on single-node, but for multi-node:

1. **Set master address:**
   ```bash
   export MASTER_ADDR="<master_node_ip>"
   export MASTER_PORT="29500"
   ```

2. **Use correct rank/world_size:**
   - Rank should be unique across all nodes
   - World size = total GPUs across all nodes

3. **Network configuration:**
   - Ensure nodes can communicate
   - Use high-bandwidth interconnect (InfiniBand)

## Debugging Tips

### 1. Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Add Checkpoints

```python
trainer.train(
    save_checkpoints=True,
    checkpoint_interval=100,
)
```

### 3. Profile Communication

```python
from src.benchmark.communication import CommunicationProfiler

profiler = CommunicationProfiler(enabled=True)
# Run training
stats = profiler.get_stats()
print(stats.to_dict())
```

### 4. Monitor Resources

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# CPU/Memory monitoring
htop

# Network monitoring (if multi-node)
iftop
```

### 5. Minimal Reproduction

Create minimal test case:
```python
# Minimal model
model = SimpleModel(hidden_size=128, num_layers=2)

# Single step
trainer = DDPTrainer(model, config, device, rank=0, world_size=1)
trainer.train(num_steps=1, warmup_steps=0)
```

## Getting Help

1. **Check logs:**
   - Look for error messages
   - Check NCCL debug output if enabled

2. **Reproduce with minimal config:**
   - Small model
   - Single GPU
   - Few steps

3. **Collect system info:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   nvidia-smi
   python scripts/train_ddp.py --help  # If implemented
   ```

4. **Check PyTorch version:**
   - Ensure PyTorch 2.0+ for FSDP
   - Check CUDA compatibility

## Prevention

1. **Always validate configs** before training
2. **Start with small models** and scale up
3. **Use checkpoints** to resume from failures
4. **Monitor resources** during training
5. **Test on single GPU** before multi-GPU
6. **Keep dependencies updated** but stable

