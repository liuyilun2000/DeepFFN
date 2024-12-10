#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --mail-type=ALL


config='768_4_4-3'

config="768_6_4-2"

config="768_12_4-1"

for dataset in "boolq" "piqa" "social_i_qa" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa" "hellaswag"; do
    python commonsense_evaluate.py \
        --dataset $dataset \
        --base_model "checkpoints/commonsense/${config}" \
        --name "commonsense-${config}" \
        --batch_size 16 --max_new_tokens 4 \
        | tee -a log/commonsense/${config}.eval.${dataset}.log
done
