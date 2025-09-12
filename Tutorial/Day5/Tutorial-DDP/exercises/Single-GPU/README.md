# Single-GPU Training

## Folder structure

- `main.py` – Main training loop
- `data.py` – Data loading utilities
- `model.py` – Model definition and evaluation function

## Exercise

Your task is to complete the training loop in `main.py` by filling in the `TODO` sections. Specifically, implement the following core steps:

- [ ] Reset the optimizer’s gradients
- [ ] Perform the forward pass
- [ ] Compute the loss
- [ ] Backpropagate the gradients
- [ ] Update model parameters (optimization step**

**Hint**: You can refer to the example notebooks in the `01-notebooks/` folder:
- `00-pytorch.ipynb`
- `01-mnist-training.ipynb`

Also, review the function definitions in `data.py` to properly retrieve the training and test DataLoaders.


## Important: Working with Devices

One important detail we haven’t covered yet is that PyTorch tensors can reside either on the host (RAM) or on a device (VRAM, such as a GPU).

To handle this, we typically define a `device` variable to indicate where the tensors should live—either `"cpu"` or `"cuda"` (GPU):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Once defined, you can create tensors directly on that device or move existing ones between the host and device. Under the hood, this triggers an asynchronous memory copy operation.

For example, to move an entire model to the selected device:

```python
model = MLP().to(device)
```

**Important:** You cannot perform operations between tensors on different devices (e.g., one on CPU and one on GPU). Make sure all involved tensors are on the same device before performing any computation.

##  Running the Code

To run the training script on the **Leonardo cluster**, use the provided submission script:

```bash
$ sbatch -A {{accout}}submit.sh
```
We will use the partition: `boost_usr_prod` and since this is for debugging and expected to run quickly, we’ll skip the queue using the `--qos=boost_qos_dbg` flag.

## Modules

The necessary software stack will be provided loading the modules in the job file:
```
module load profile/deeplrn
module load cineca-ai/
```
