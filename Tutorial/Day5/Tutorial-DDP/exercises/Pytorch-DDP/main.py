import torch
import time
import os
import sys
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from model import AlexNetCIFAR,evaluate
from data import get_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

assert dist.is_available()
dist.init_process_group("nccl")
rank=dist.get_rank()
world_size=dist.get_world_size()
device_id = rank % torch.cuda.device_count()
print(f"Hello from rank {rank}, using device: {device_id}\n")

# Wrap the model using Distributed Data Parallel ! 
model = AlexNetCIFAR().to(device_id)
ddp_model = DDP(model, device_ids=[device_id])

train_dataset,test_dataset = get_dataset()

# Distributed sampler, it is initialized with rank and world_size. It distribute the index of the training sample to all the processes.
train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
test_sampler = DistributedSampler(test_dataset,num_replicas=world_size,rank=rank)

train_loader = DataLoader(train_dataset, shuffle=False,
                              sampler=train_sampler,batch_size=1024//world_size,num_workers=2, drop_last=True,pin_memory=True)

test_loader = DataLoader(test_dataset, shuffle=False,
                             sampler=test_sampler, drop_last=True)

# Define the cross entropy loss
loss = torch.nn.CrossEntropyLoss()
# Use the adam optimizer
optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    # Sync the distributed sampler achievi a suffle
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    start_time = time.time()
    ddp_model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device_id), targets.to(device_id)
        optimizer.zero_grad() 
        outputs = ddp_model(inputs)
        loss_value = loss(outputs, targets)
        loss_value.backward()
        optimizer.step()

    walltime= time.time() - start_time
    # Optional: Evaluation (note this can be computationally expensive)
    correct, total = evaluate(ddp_model, test_loader, device_id)
    
    if rank==0:
        print(f'Epoch {epoch}, Accuracy {correct/total}, Walltime per epoch: {walltime:.4f}s')
