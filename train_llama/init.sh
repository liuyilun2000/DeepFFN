#!/bin/bash

# HuggingFace token
HF_TOKEN="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"

# Base directory for all models
BASE_DIR="./DeepFFNLLaMA"

# Function to initialize a model with given parameters
init_model() {
    local hidden_size=$1
    local num_layers=$2
    local num_heads=$3
    local intermediate_ratio=$4
    local mlp_layers=$5
    
    # Construct output directory name (without heads in the name)
    local model_name="DeepFFNLLaMA_${hidden_size}_${num_layers}_${intermediate_ratio}-${mlp_layers}"
    local output_dir="${BASE_DIR}/${model_name}"
    
    echo "Initializing model: ${model_name}"
    echo "Configuration: hidden_size=${hidden_size}, layers=${num_layers}, heads=${num_heads}, intermediate_ratio=${intermediate_ratio}, mlp_layers=${mlp_layers}"
    echo "Output directory: ${output_dir}"
    
    python init.py \
        --hidden-size ${hidden_size} \
        --intermediate-size-ratio ${intermediate_ratio} \
        --num-hidden-layers ${num_layers} \
        --num-attention-heads ${num_heads} \
        --num-mlp-layers ${mlp_layers} \
        --output-dir ${output_dir} \
        --bf16 \
        --hf-token ${HF_TOKEN} \
        --calc-non-emb-params
        
    echo "Model initialized for: ${model_name}"
    echo "----------------------------------------"
}

# Create base directory if it doesn't exist
mkdir -p ${BASE_DIR}

# Configuration array
# Format: "hidden_size layers attention_heads intermediate_ratio mlp_layers"
'''
configs=(
    "768 12 12 4 1"
    "768 12 12 2 2"
    "768 12 12 1 4"
    "1024 24 16 4 1"
    "1024 24 16 2 2"
    "1024 24 16 1 4"
)
'''
configs=(
    "768 6 12 4 2"
    "768 4 12 4 3"
    "768 3 12 4 4"
)

# Initialize all configurations
for config in "${configs[@]}"; do
    read -r hidden_size layers heads intermediate_ratio mlp_layers <<< "$config"
    init_model "$hidden_size" "$layers" "$heads" "$intermediate_ratio" "$mlp_layers"
done

echo "All models initialized successfully!"