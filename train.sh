python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_2-2 \
  --output-dir ./output/768_12_2-2 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_2-2" \
  --resume_from_checkpoint "checkpoint-64"