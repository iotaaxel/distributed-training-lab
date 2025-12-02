#!/bin/bash
# Convenience script to run DDP training on a single node

NUM_GPUS=${1:-4}  # Default to 4 GPUs if not specified

echo "Running DDP training with $NUM_GPUS GPUs..."
torchrun --nproc_per_node=$NUM_GPUS src/ddp_train.py --config configs/ddp_resnet_cifar10.yaml

