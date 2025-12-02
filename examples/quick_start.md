# Quick Start Guide

This guide walks you through running your first distributed training experiments.

## Prerequisites

- 1-4 NVIDIA GPUs with CUDA support
- PyTorch 2.0+ with CUDA
- Python 3.8+

## Step 1: Verify Setup

Check that PyTorch can see your GPUs:

```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Step 2: Run DDP Training

Start with DDP on a single GPU:

```bash
python scripts/train_ddp.py
```

You should see output like:
```
DDP Training
==================================================
World size: 1
Device: cuda:0
...
Warming up for 10 steps...
Benchmarking for 50 steps...
Step 10/60 | Loss: 9.2341 | Time: 45.23ms
...
```

## Step 3: Run Multi-GPU DDP

If you have multiple GPUs, try:

```bash
torchrun --nproc_per_node=2 scripts/train_ddp.py
```

This will use 2 GPUs. Adjust `--nproc_per_node` to match your GPU count.

## Step 4: Run FSDP Training

Try FSDP on a single GPU:

```bash
python scripts/train_fsdp.py
```

Then try multi-GPU:

```bash
torchrun --nproc_per_node=2 scripts/train_fsdp.py
```

## Step 5: Compare Performance

Run the benchmark script to compare DDP vs FSDP:

```bash
torchrun --nproc_per_node=2 scripts/benchmark.py
```

This will:
1. Train with DDP and collect metrics
2. Train with FSDP and collect metrics
3. Print a comparison table
4. Save results to `results/benchmark_results.json`

## Step 6: Experiment with Configs

Edit `configs/ddp_config.yaml` to change model size:

```yaml
model:
  hidden_size: 2048  # Increase from 1024
  num_layers: 24      # Increase from 12
```

Then re-run to see how performance changes.

## Troubleshooting

**Issue**: `NCCL error` or `CUDA out of memory`
- **Solution**: Reduce `batch_size` or model size in config files

**Issue**: Script hangs on multi-GPU
- **Solution**: Ensure all GPUs are accessible and not in use by other processes

**Issue**: `torchrun` not found
- **Solution**: Update PyTorch: `pip install --upgrade torch`

## Next Steps

- Read the main README for detailed explanations
- Experiment with different model sizes
- Try enabling mixed precision in FSDP
- Profile training with `profile=True`

