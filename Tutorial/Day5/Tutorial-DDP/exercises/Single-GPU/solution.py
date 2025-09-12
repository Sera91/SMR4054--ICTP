from data import get_dataloader
from model import MLP,evaluate
from time import time
import torch

# Check if CUDA devices are available; if yes, move everything to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Instantiate the model and then move it to the selected device.
model = MLP().to(device)

# Set up the SGD optimizer and pass the model parameters to it.
# It will take care of updating them during training.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Set up the loss function — use cross-entropy loss for classification tasks.
loss_fn = torch.nn.CrossEntropyLoss()



# Use the function get_data() (defined in data.py) to get train_loader and test_loader.
# Retrieve data loaders
train_loader,test_loader = get_dataloader()
for epoch in range(20):
    # Set model to training mode: it enables gradient calculation and dropout (if used).
    model.train()
    start = time()
    loss_epoch = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.view(batch_x.size(0), -1)  # flatten
        # Move data to GPU !
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Zero out gradients in the optimizer
        
        optimizer.zero_grad()
        # Forward pass — evaluate the input to get model predictions
        y_predicted = model(batch_x.to(device))
        # Compute loss comparing predictions and true labels
        l = loss_fn(y_predicted, batch_y.to(device))
        # Backward pass — compute gradients
        l.backward()
        # Optimization step — update model parameters
        optimizer.step()
        loss_epoch += l.item()
    end = time()-start

    # Calculate accuracy using the `evaluate()` function defined in `model.py`
    accuracy = evaluate(model, test_loader,device=device)
    print(f"Epoch {epoch}, Loss: {loss_epoch:.4f}, Accuracy: {accuracy:.4f}, Epoch walltime: {end:.5f}")
