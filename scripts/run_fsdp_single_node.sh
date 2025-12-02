#!/bin/bash
# Convenience script to run FSDP training on a single node

NUM_GPUS=${1:-4}  # Default to 4 GPUs if not specified

echo "Running FSDP training with $NUM_GPUS GPUs..."
torchrun --nproc_per_node=$NUM_GPUS src/fsdp_train.py --config configs/fsdp_resnet_cifar10.yaml

