#!/bin/bash
module purge
module load cuDNN virtualenv

cd /home/serafina.di-gioia/lustre/training_t-7bpcjwtiufg/users/serafina.di-gioia/

python3 -m venv Seraenv/

source Seraenv/bin/activate
pip install wheel
pip install torchvision torch matplotlib seaborn 
pip install ipython


echo "import torchvision
torchvision.datasets.MNIST(root='./data', train=True, download=True)
torchvision.datasets.MNIST(root='./data', train=False, download=True)
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)" > main.py
python main.py
rm -f main.py
module purge
