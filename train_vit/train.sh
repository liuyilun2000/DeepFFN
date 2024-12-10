#!/bin/bash

python train.py \
    --model-dir ./DeepFFNViT/DeepFFNViT_768_6_4-2 \
    --data-dir ../train_vit/data/imagenet \
    --output-dir ./output/DeepFFNViT_768_6_4-2 \
    --batch-size 32 \
    --grad-accum 4 \
    --lr 1e-3 \
    --wandb-project deepffn-vit \
    --wandb-name DeepFFNViT_768_6_4-2 \
    --bf16 \
    --cpu \
    | tee 768_6_4-2.log
