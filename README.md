# Distributed Training Lab

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              ██████╗ ██╗███████╗████████╗                   ║
║              ██╔══██╗██║██╔════╝╚══██╔══╝                   ║
║              ██║  ██║██║███████╗   ██║                      ║
║              ██║  ██║██║╚════██║   ██║                      ║
║              ██████╔╝██║███████║   ██║                      ║
║              ╚═════╝ ╚═╝╚══════╝   ╚═╝                      ║
║                                                              ║
║         ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ║
║         ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║ ║
║            ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║ ║
║            ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║ ║
║            ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║ ║
║            ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ║
║                                                              ║
║         DDP • FSDP • Benchmarking • Single-Node              ║
╚══════════════════════════════════════════════════════════════╝
```

**Minimal PyTorch DDP/FSDP lab for benchmarking distributed training on a single node.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)](https://github.com/psf/black)

</div>

---

## Motivation

This repo exists because I care deeply about PyTorch internals and performance. It's a clean reference implementation for both interview discussions and real production workloads. Everything is designed to run on a single machine with 1-4 GPUs, making it accessible while demonstrating production-quality distributed training patterns.

## Features

- **Single-node multi-GPU DDP example** - Full model replication with gradient synchronization
- **Single-node FSDP example** - Parameter sharding with configurable strategies
- **Simple CIFAR-10 dataset** - Fast iteration, no huge downloads
- **Benchmark script** - Compares DDP vs FSDP on:
  - Samples/sec throughput
  - Average step time
  - Peak GPU memory usage

## Quickstart

```bash
git clone https://github.com/yourusername/distributed-training-lab.git
cd distributed-training-lab
pip install -r requirements.txt
```

### DDP (single node, 4 GPUs)

```bash
torchrun --nproc_per_node=4 src/ddp_train.py --config configs/ddp_resnet_cifar10.yaml
```

### FSDP

```bash
torchrun --nproc_per_node=4 src/fsdp_train.py --config configs/fsdp_resnet_cifar10.yaml
```

### Benchmark both

```bash
python src/benchmark_runner.py
```

Or use the convenience scripts:

```bash
bash scripts/run_ddp_single_node.sh
bash scripts/run_fsdp_single_node.sh
```

## Configuration

YAML configs control model architecture, batch size, learning rate, optimizer, and FSDP sharding strategy. Key settings:

- **Model**: Small ResNet (configurable depth/width)
- **Batch size**: Per-GPU batch size (effective batch = batch_size × num_gpus)
- **Precision**: FP32 by default, FP16 available for FSDP
- **Sharding**: FULL_SHARD (default) or SHARD_GRAD_OP for FSDP

Example config structure:
```yaml
model:
  name: "small_resnet"
  num_layers: 18
  
training:
  batch_size: 128
  num_epochs: 10
  learning_rate: 0.1
  optimizer: "sgd"
```

## What This Demonstrates

- **DDP replication vs FSDP sharding**: DDP keeps full model on each GPU; FSDP splits parameters across GPUs, trading memory for communication overhead
- **Batch size scaling**: Effective batch size = per_gpu_batch × num_gpus; larger batches improve GPU utilization but require more memory
- **Memory vs communication tradeoff**: DDP uses more memory but less communication; FSDP saves memory but adds all-gather overhead
- **DistributedSampler usage**: Ensures each GPU sees different data partitions, critical for correct distributed training
- **Gradient synchronization**: DDP uses allreduce (one op per step); FSDP uses allgather + allreduce (more ops, but sharded)
- **Single-node patterns**: How to structure code that works on 1 GPU (testing) and scales to 4 GPUs (production)

## Limitations / Future Work

- No multi-node launcher yet (single-node focus keeps it simple)
- Could add mixed precision variants, gradient checkpointing, larger models
- Pipeline/tensor parallelism not included (keeps scope focused)

## Repository Structure

```
distributed-training-lab/
├── README.md
├── requirements.txt
├── configs/
│   ├── ddp_resnet_cifar10.yaml
│   └── fsdp_resnet_cifar10.yaml
├── src/
│   ├── data.py              # CIFAR-10 dataset + dataloaders
│   ├── models.py            # Small ResNet implementation
│   ├── utils.py             # Logging, timing, seeding
│   ├── ddp_train.py         # DDP training entrypoint
│   ├── fsdp_train.py        # FSDP training entrypoint
│   └── benchmark_runner.py  # Runs both, prints comparison table
├── scripts/
│   ├── run_ddp_single_node.sh
│   └── run_fsdp_single_node.sh
└── examples/
    └── notes_ddp_vs_fsdp.md  # Conceptual notes
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
