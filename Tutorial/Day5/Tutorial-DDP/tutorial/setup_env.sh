#!/bin/bash
module purge
module load cuDNN virtualenv

#to be modified with your username
cd /home/$USER/lustre/training_t-7bpcjwtiufg/users/$USER/


python3 -m venv Seraenv/

source Seraenv/bin/activate
pip install wheel
pip install torchvision torch matplotlib seaborn 
pip install ipython
