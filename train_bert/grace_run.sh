#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH -J difan
#SBATCH -o ../../gracelogs/deepffn/deepffnroberta.%J.out
#SBATCH -e ../../gracelogs/deepffn/deepffnroberta.%J.err
#SBATCH --time=360:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

python train.py \
  --model-dir ./DeepFFNRoBERTa/DeepFFNRoberta_768_12_4-1 \
  --output-dir ./output/768_12_4-1 \
  --lr 6e-4  --bf16 \
  --micro-batch-size 32 \
  --wandb-project "deepffn" \
  --wandb-run "roberta-openwebtext-768_12_4-1" \
  # --resume_from_checkpoint "checkpoint-630"
