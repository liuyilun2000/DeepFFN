#!/bin/bash

# Check if .env file exists in home directory
if [ ! -f "${HOME}/.env" ]; then
    echo "Error: .env file not found in home directory"
    echo "Please create ${HOME}/.env with your HuggingFace token like this:"
    echo "HF_TOKEN=your_token_here"
    exit 1
fi

# Load HF token from .env file in home directory
source "${HOME}/.env"

# Validate that token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found in ${HOME}/.env file"
    exit 1
fi

# Base directory for all models
BASE_DIR="./DroppedLLaMA"

# Function to initialize a model with given parameters
init_model() {
    local hidden_size=$1
    local num_layers=$2
    local num_heads=$3
    local intermediate_ratio=$4
    local attention_layers=$5
    
    # Construct output directory name (without heads in the name)
    local attn_pattern=$(echo $attention_layers | tr ',' '_')
    local model_name="DroppedLLaMA_${hidden_size}_${num_layers}_${intermediate_ratio}-attn_${attn_pattern}"
    local output_dir="${BASE_DIR}/${model_name}"
    
    echo "Initializing model: ${model_name}"
    echo "Configuration: hidden_size=${hidden_size}, layers=${num_layers}, heads=${num_heads}, intermediate_ratio=${intermediate_ratio}, attention_layers=${attention_layers}"
    echo "Output directory: ${output_dir}"
    
    python init.py \
        --hidden-size ${hidden_size} \
        --intermediate-size-ratio ${intermediate_ratio} \
        --num-hidden-layers ${num_layers} \
        --num-attention-heads ${num_heads} \
        --attention-layers "${attention_layers}" \
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
# Format: "hidden_size layers attention_heads intermediate_ratio attention_layers"

configs=(
    "768 12 12 4 0,1,2,3,4,5,6,7,8,9,10,11"
    "768 12 12 4 0,1,4,7,10,11"
    "768 12 12 4 0,2,4,6,8,10"
    "768 12 12 4 0,1,2,9,10,11"
    "768 12 12 4 0,1,2,4,7,9,10,11"
    "768 12 12 4 0,1,2,5,6,9,10,11"
)

# Initialize all configurations
for config in "${configs[@]}"; do
    read -r hidden_size layers heads intermediate_ratio attention_layers <<< "$config"
    init_model "$hidden_size" "$layers" "$heads" "$intermediate_ratio" "$attention_layers"
done

echo "All models initialized successfully!"
