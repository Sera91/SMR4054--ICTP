#!/bin/bash
#SBATCH --job-name=single-gpu
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu_h100
#SBATCH --output=slurm.out
#SBATCH --account=TRAINING_T-7BPCJWTIUFG-PREMIUM-GPU

#####################################
#       ENV SETUP                   #
#####################################
module purge
module load cuDNN virtualenv
cd /home/serafina.di-gioia/lustre/training_t-7bpcjwtiufg/users/serafina.di-gioia/

python3 -m venv Seraenv/
source Seraenv/bin/activate

cd Tutorial-DDP/tutorial/

export OMP_NUM_THREADS=1

#####################################
#       RESOURCES                   #
#####################################
echo "Node allocated ${SLURM_NODELIST}"
echo "Requested ${SLURM_NNODES} nodes"
echo "Requested ${SLURM_NTASKS} tasks in total"
echo "Requested ${SLURM_TASKS_PER_NODE} task per node"
echo ""

echo "Requested ${SLURM_GPUS_ON_NODE} gpus per node"
#echo "Total gpu requested ${SLURM_GPUS}"
###################################

export LOGLEVEL=INFO

python single_gpu.py 50 10

