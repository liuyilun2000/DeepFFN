#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=7:59:00
#SBATCH --mail-type=ALL


config="768_12_4-1"
#config="768_6_4-2"
#config="768_4_4-3"


torchrun \
  --nproc_per_node 4 train.py \
    --model-dir "./DeepFFNLLaMA/DeepFFNLLaMA_${config}" \
    --output-dir "./output/${config}" \
    --lr 5e-4  --bf16 \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --wandb-project "deepffn" \
    --wandb-run "${config}"
