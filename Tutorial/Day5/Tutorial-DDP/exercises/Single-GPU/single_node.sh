#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
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

cd Tutorial-DDP/exercises/

export LOGLEVEL=INFO
#####################################
#       RESOURCES                   #
#####################################
echo "Node allocated ${SLURM_NODELIST}"
echo "Requested ${SLURM_GPUS_ON_NODE} gpus per node"
####################################
#      MASTER ELECTION             #
####################################
# Work just with newer slurm version!
#export MASTER_ADDR=$(scontrol getaddrs $SLURM_NODELIST | head -n1 | awk -F ':' '{print$2}' | sed 's/^[ \t]*//;s/[ \t]*$//')
####################################
export MASTER_ADDR=$( ip -4 addr show enp1s0f0 | awk '/inet / {print $2}' | cut -d/ -f1)
export MASTER_PORT=12345
echo "Master's ip address used ${MASTER_ADDR}"
export RANDOM=42

export LOGLEVEL=INFO

torchrun \
--nnodes 1 \
--nproc_per_node 2 main.py
#--rdzv_id ${RANDOM} \
#--rdzv_backend c10d \
#--rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} main.py

