#!/bin/bash
module purge
module load cuDNN virtualenv
module load Autotools/20220317-GCCcore-11.3.0
module load netCDF-Fortran/4.6.0-iimpi-2022a

#sourcing virtual environment
source /home/$USER/lustre/training_t-7bpcjwtiufg/shared/Tutenv/bin/activate

#entering the local repo for the tutorial

cd /home/$USER/lustre/training_t-7bpcjwtiufg/users/$USER/SMR4054--ICTP/Tutorial/Day7/RCM-emulation-tutorial/data/

python data_download.py $USER
                                    
