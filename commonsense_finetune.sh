#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --job-name=commonsense_finetune
#SBATCH --output=log/commonsense_finetune.slurm.log
#SBATCH --mail-type=ALL

model_name="steph0713/deepffnllama-768_12_4-1"
config_name="768_12_4-1"

model_name="steph0713/deepffnllama-768_6_4-2"
config_name="768_6_4-2"

model_name="steph0713/deepffnllama-768_4_4-3"
config_name="768_4_4-3"


python finetune.py \
    --base_model "${model_name}" \
    --data_path 'commonsense_170k.json' \
    --output_dir "checkpoints/commonsense/${config_name}" \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 1e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --wandb_project "deepffn" \
    --wandb_run_name "commonsense_${config_name}" \
    2>&1 | tee -a "log/commonsense/${config_name}.log"

echo "All training jobs completed."