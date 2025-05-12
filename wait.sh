#!/bin/bash

#SBATCH --job-name=__
#SBATCH --time=4-00:00:00
#SBATCH --output=%j_%x_%a.out
#SBATCH --partition=batch_grad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH --gres=gpu:4

# Your job commands go below this line
echo "Hello, this is a Slurm job script!"
# Sleep for sbatch timelimit
sleep $((3600 * 24 * 7))