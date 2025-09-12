import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # input layer to hidden layer
        self.fc2 = nn.Linear(128, 64)     # hidden to hidden
        self.fc3 = nn.Linear(64, 10)      # hidden to output (10 classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)         # flatten input (batch_size, 784)
        x = F.relu(self.fc1(x))           # first hidden layer + ReLU
        x = F.relu(self.fc2(x))           # second hidden layer + ReLU
        x = self.fc3(x)                   # output logits
        return x

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
