#!/bin/bash
#SBATCH --job-name=run_training
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu_h100
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --account=TRAINING_T-7BPCJWTIUFG-PREMIUM-GPU

#####################################
#       ENV SETUP                   #
#####################################
module purge
module load cuDNN virtualenv
module load Autotools/20220317-GCCcore-11.3.0
module load netCDF-Fortran/4.6.0-iimpi-2022a

#sourcing virtual environment
source /home/$USER/lustre/training_t-7bpcjwtiufg/shared/Tutenv/bin/activate

#entering the local repo for the tutorial

cd /home/$USER/lustre/training_t-7bpcjwtiufg/$USER/SMR4054--ICTP/Tutorial/Day7/RCM-emulation-tutorial/dl-model/

python training_evaluation.py $USER
~                                    
