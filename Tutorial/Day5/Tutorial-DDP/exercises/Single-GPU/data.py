import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(batch_size=128):

    # 1. Normalize MNIST: mean=0.1307, std=0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Download and transform MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=False)

    # 3. Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader
