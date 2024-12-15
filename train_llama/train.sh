#!/bin/bash

HIDDEN_SIZE=768
NUM_LAYERS=12
INTER_RATIO=4

# ATTN_LAYERS="0_1_2_3_4_5_6_7_8_9_10_11"
# ATTN_LAYERS="0_1_2_5_6_9_10_11"
# ATTN_LAYERS="0_1_2_9_10_11"
# ATTN_LAYERS="0_1_4_7_10_11"
# ATTN_LAYERS="0_2_4_6_8_10"
ATTN_LAYERS="0_1_2_4_7_9_10_11"

MODEL_NAME="DroppedLLaMA_${HIDDEN_SIZE}_${NUM_LAYERS}_${INTER_RATIO}-attn_${ATTN_LAYERS}"
WANDB_RUN="llama-${HIDDEN_SIZE}-${NUM_LAYERS}-${INTER_RATIO}-attn${ATTN_LAYERS}"
OUTPUT_DIR="./output/${HIDDEN_SIZE}_${NUM_LAYERS}_${INTER_RATIO}-attn${ATTN_LAYERS}"

echo "Starting training for model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"

python train.py \
  --model-dir "./DroppedLLaMA/$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --lr 6e-4 \
  --bf16 \
  --micro-batch-size 32 \
  --wandb-project "dropped_llama" \
  --wandb-run "$WANDB_RUN" \

torchrun \
  --nproc_per_node 4 train.py \
    --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_6_4-2 \
    --output-dir ./output/768_6_4-2 \
    --lr 5e-4  --bf16 \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --wandb-project "deepffn" \
    --wandb-run "768_6_4-2"
