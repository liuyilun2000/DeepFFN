python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_2-2 \
  --output-dir ./output/768_12_2-2 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_2-2" \
  --resume_from_checkpoint "checkpoint-190"



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_4-1 \
  --output-dir ./output/768_12_4-1 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_4-1" \
  --resume_from_checkpoint "checkpoint-200"



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_1-4 \
  --output-dir ./output/768_12_1-4 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_1-4" 


  

python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_6_4-2 \
  --output-dir ./output/768_6_4-2 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_6_4-2" 



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_4_4-3 \
  --output-dir ./output/768_4_4-3 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_4_4-3" 



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_3_4-4 \
  --output-dir ./output/768_3_4-4 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_3_4-4" 
