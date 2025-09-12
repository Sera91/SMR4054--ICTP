#!/bin/bash
module purge
module load cuDNN virtualenv

#to be modified with your username
cd /home/serafina.di-gioia/lustre/training_t-7bpcjwtiufg/users/serafina.di-gioia/


python3 -m venv Seraenv/

source Seraenv/bin/activate
pip install wheel
pip install torchvision torch matplotlib seaborn 
pip install ipython
