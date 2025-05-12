#!/bin/bash

#SBATCH -J BPM_MT
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=36G
#SBATCH -o %x_%j_%a.out
#SBATCH -e %x_%j_%a.err
#SBATCH --time=4-00:00:00

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=$SLURM_NNODES

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /data/moonjunyyy/init.sh
conda activate BC

conda --version
python --version

model=${1}
mode=${2}
dataset=${3}

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0;9.0a"

# seeds=(1 21 42 3473 10741 32450 93462 85015 64648 71950 87557 99668 55552 4811 10741)
seeds=(99668)

for i in ${seeds[@]}
do
    echo "seed: $i"
    python main.py \
    --model ${model} \
    --mode ${mode} \
    --dataset ${dataset} \
    --seed ${i} \
    --batch_size 128 \
    --num_workers 8 \
    --epochs 100 \
    --language koBert \
    --audio HuBert \
    --video VideoMAE \
    --lr 0.0005 \
    --dropout 0.3 \
    --world_size $WORLD_SIZE \
    --rank $SLURM_PROCID ${@:4}
done