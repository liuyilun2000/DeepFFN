#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH -J difan
#SBATCH --time=10:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

'''
# Check if the model type parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Model type parameter is required."
    echo "Usage: sbatch $0 <model_type>"
    exit 1
fi

# Get the model type from the command line argument
MODEL_TYPE=$1

'''
declare -a configs=(
    "768_12_4-1"
    #"768_6_4-2"
    #"768_4_4-3"
    #"768_12_3-1"
)

# Function to run evaluation
run_evaluation() {
    local config=$1
    local gpu_id=$2
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for dataset in "boolq" "piqa" "social_i_qa" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa" "hellaswag"; do
        python commonsense_evaluate.py \
            --dataset $dataset \
            --base_model ${config} \
            --name ${config} \
            --batch_size 16 --max_new_tokens 4 \
            | tee -a commonsense/log/${config}.eval.${dataset}.log
    done
}

# Run evaluations in parallel
for i in {0..3}; do
    run_evaluation "${configs[$i]}" $i &
done

# Wait for all background jobs to finish
wait