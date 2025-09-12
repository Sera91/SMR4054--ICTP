# Distributed Data Parallel (DDP)

**Distributed Data Parallel (DDP)** is a built-in PyTorch utility that wraps your model for multi-GPU training.  

It allows you to scale your single-GPU training code to multiple GPUs by adding just a few lines of code (3 L.O.C and you can scale to hundreds GPUs)!

---

## Distributed Sampler

To ensure that each process trains on a unique subset of the data, PyTorch uses `DistributedSampler`.  
This sampler splits the dataset across the different processes (or ranks), and shuffles it independently per epoch.

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
```

---

## Model Wrapping

Once the model is on the correct GPU, wrap it using `DistributedDataParallel`.
PyTorch will automatically handle gradient synchronization and communication between GPUs.

```python
from torch.nn.parallel import DistributedDataParallel as DDP

device_id = rank % torch.cuda.device_count()
model = AlexNetCIFAR().to(device_id)
ddp_model = DDP(model, device_ids=[device_id])
```

---

## Running the Code

On the **Leonardo cluster**, you can use the `submit.sh` job script.
It will launch your training script using `torchrun`, which handles NCCL initialization and process spawning automatically.

Example:

```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE your_script.py
```

The submist script has many more flags, that enable us to scale to multiple nodes. See `00-mutinode` folder to get more hints.
