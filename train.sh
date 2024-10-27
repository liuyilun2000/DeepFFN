python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_2-2 \
  --output-dir ./output/768_12_2-2 \
  --lr 1e-3  --bf16 \
  --micro-batch-size 30 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_2-2" \
  --resume_from_checkpoint "checkpoint-620"



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_4-1 \
  --output-dir ./output/768_12_4-1 \
  --lr 1e-3  --bf16 \
  --micro-batch-size 30 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_4-1" \
  --resume_from_checkpoint "checkpoint-580"



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_1-4 \
  --output-dir ./output/768_12_1-4 \
  --lr 1e-3  --bf16 \
  --micro-batch-size 30 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_1-4"  \
  --resume_from_checkpoint "checkpoint-160"


  

python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_6_4-2 \
  --output-dir ./output/768_6_4-2 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_6_4-2" \
  --resume_from_checkpoint "checkpoint-560"




python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_4_4-3 \
  --output-dir ./output/768_4_4-3 \
  --lr 1e-3  --bf16 \
  --micro-batch-size 30 \
  --wandb-project "deepffn" \
  --wandb-run "768_4_4-3" \
  --resume_from_checkpoint "checkpoint-620"














python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_3_4-4 \
  --output-dir ./output/768_3_4-4 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_3_4-4" 
