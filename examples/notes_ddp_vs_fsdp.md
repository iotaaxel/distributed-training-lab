# DDP vs FSDP: Conceptual Notes

## Key Differences

### DDP (DistributedDataParallel)

**How it works:**
- Each GPU holds a **full copy** of the model
- During backward pass, gradients are **allreduced** (averaged) across GPUs
- Each GPU then updates its own copy of parameters

**Memory:** `N_GPUs × (Model + Gradients + Optimizer States)`

**Communication:** One allreduce per step (gradients only)

**Best for:** Models that fit comfortably on a single GPU

### FSDP (Fully Sharded Data Parallel)

**How it works:**
- Model parameters are **sharded** across GPUs (each GPU holds 1/N)
- During forward: **allgather** sharded params → compute → discard
- During backward: **allgather** params → compute gradients → **allreduce** gradients → discard
- Optimizer states are also sharded

**Memory:** `(Model + Gradients + Optimizer States) / N_GPUs + temporary buffers`

**Communication:** Multiple allgathers + allreduce per step

**Best for:** Models that exceed single GPU memory

## Tradeoffs

| Aspect | DDP | FSDP |
|--------|-----|------|
| Memory efficiency | Lower (full replication) | Higher (sharding) |
| Communication overhead | Lower (one allreduce) | Higher (allgathers + allreduce) |
| Model size limit | Single GPU memory | Can exceed single GPU |
| Throughput (small models) | Higher | Lower |
| Throughput (large models) | N/A (OOM) | Only option |

## When to Use Which?

- **Model fits on 1 GPU** → Use DDP (better performance)
- **Model exceeds 1 GPU** → Use FSDP (only option)
- **Need maximum throughput** → DDP (if model fits)
- **Need maximum memory efficiency** → FSDP

## Communication Bottlenecks

**DDP bottlenecks:**
- Allreduce time scales with model size
- Interconnect bandwidth (NVLink > PCIe)

**FSDP bottlenecks:**
- Allgather operations (more frequent than DDP)
- Network topology matters more
- Can overlap communication with computation (advanced)

## Memory Breakdown

For a typical model:
- **Parameters**: ~100M params = ~400 MB (FP32)
- **Gradients**: Same size as parameters = ~400 MB
- **Optimizer states** (Adam): 2× parameters = ~800 MB
- **Activations**: Depends on batch size and model architecture

**DDP on 4 GPUs:**
- Per GPU: 400 + 400 + 800 + activations = ~1.6 GB + activations
- Total: 4 × 1.6 GB = 6.4 GB (plus activations)

**FSDP on 4 GPUs:**
- Per GPU: (400 + 400 + 800) / 4 + activations = ~400 MB + activations
- Total: ~1.6 GB + 4 × activations

## Interview Talking Points

1. **DistributedSampler**: "We use DistributedSampler to ensure each GPU sees different data. Without it, all GPUs would process the same batches, defeating the purpose of data parallelism."

2. **Gradient synchronization**: "DDP uses allreduce to average gradients. This happens automatically during backward() - PyTorch hooks into the backward pass."

3. **FSDP sharding**: "FSDP shards at the module level. We wrap BasicBlocks individually, which gives us better memory efficiency than wrapping the entire model."

4. **Communication vs compute**: "For small models, DDP is faster because communication overhead is low. For large models, FSDP is necessary even though it has higher communication costs."

5. **Memory efficiency**: "FSDP allows us to train models 4x larger than single GPU memory on 4 GPUs, but we pay with 2-3x more communication operations per step."

