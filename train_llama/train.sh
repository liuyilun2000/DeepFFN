python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_2-2 \
  --output-dir ./output/768_12_2-2 \
  --lr 1e-3  --bf16 \
  --per_device_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_2-2" 


python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_1-4 \
  --output-dir ./output/768_12_1-4 \
  --lr 1e-3  --bf16 \
  --per_device_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_1-4" 






python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_12_4-1 \
  --output-dir ./output/768_12_4-1 \
  --lr 5e-4  --bf16 \
  --per_device_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --wandb-project "deepffn" \
  --wandb-run "768_12_4-1"

  

python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_6_4-2 \
  --output-dir ./output/768_6_4-2 \
  --lr 5e-4  --bf16 \
  --per_device_batch_size 32 \
  --gradient_accumulation_steps 16 \
  --wandb-project "deepffn" \
  --wandb-run "768_6_4-2"



python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_4_4-3 \
  --output-dir ./output/768_4_4-3 \
  --lr 1e-4  --bf16 \
  --per_device_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --wandb-project "deepffn" \
  --wandb-run "768_4_4-3"












python train.py \
  --model-dir ./DeepFFNLLaMA/DeepFFNLLaMA_768_3_4-4 \
  --output-dir ./output/768_3_4-4 \
  --lr 1e-3  --bf16 \
  --wandb-project "deepffn" \
  --wandb-run "768_3_4-4" 
