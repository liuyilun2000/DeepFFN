import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import json
import math
import re
import sys
from os.path import join
from pathlib import Path
from typing import List, Optional, Union, Dict
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from tqdm import tqdm
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    Trainer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from DeepFFNLLaMA.configuration_llama import DeepFFNLlamaConfig
from DeepFFNLLaMA.modeling_llama import (
    DeepFFNLlamaModel,
    DeepFFNLlamaForCausalLM
)

AutoConfig.register("deepffn-llama", DeepFFNLlamaConfig)
AutoModel.register(DeepFFNLlamaConfig, DeepFFNLlamaModel)
AutoModelForCausalLM.register(DeepFFNLlamaConfig, DeepFFNLlamaForCausalLM)


from DroppedLLaMA.configuration_llama import DroppedLlamaConfig
from DroppedLLaMA.modeling_llama import (
    DroppedLlamaModel,
    DroppedLlamaForCausalLM
)

AutoConfig.register("dropped-llama", DroppedLlamaConfig)
AutoModel.register(DroppedLlamaConfig, DroppedLlamaModel)
AutoModelForCausalLM.register(DroppedLlamaConfig, DroppedLlamaForCausalLM)


from utils import *


def create_splits(dataset_name: str, cache_dir: str, val_size: int = 10000):
    print(f"Loading dataset {dataset_name}...")
    
    splits_cache_dir = os.path.join(cache_dir, "splits")
    train_cache_path = os.path.join(splits_cache_dir, "train")
    val_cache_path = os.path.join(splits_cache_dir, "validation")

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print("Loading splits from cache...")
        try:
            train_dataset = Dataset.load_from_disk(train_cache_path)
            val_dataset = Dataset.load_from_disk(val_cache_path)
            print(f"Loaded cached splits - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
            return train_dataset, val_dataset
        except Exception as e:
            print(f"Failed to load cached splits: {e}")
            print("Falling back to creating new splits...")

    print(f"Creating new validation split with {val_size} examples...")
    
    full_dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=False,
        cache_dir=cache_dir
    )
    
    full_dataset = full_dataset.shuffle(seed=42)
    splits = full_dataset.train_test_split(
        test_size=val_size,
        shuffle=False
    )
    
    train_dataset = splits['train']
    val_dataset = splits['test']

    print("Saving splits to cache...")
    os.makedirs(splits_cache_dir, exist_ok=True)
    train_dataset.save_to_disk(train_cache_path)
    val_dataset.save_to_disk(val_cache_path)
    
    print(f"Created splits - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    return train_dataset, val_dataset

def preprocess_and_cache_dataset(
    dataset: Dataset,
    cache_dir: str,
    split_name: str,
    preprocess_fn,
    num_proc: int = None,
    force_reprocess: bool = False
):
    """Preprocess dataset with multiprocessing and caching support."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"processed_{split_name}")
    
    if not force_reprocess and os.path.exists(cache_path):
        print(f"Loading preprocessed {split_name} dataset from cache...")
        try:
            return Dataset.load_from_disk(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}")
            print("Falling back to preprocessing...")
    
    if num_proc is None:
        num_proc = max(1, int(cpu_count()))
    
    print(f"Preprocessing {split_name} dataset using {num_proc} processes...")
    processed_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"Preprocessing {split_name} split"
    )
    
    print(f"Saving preprocessed {split_name} dataset to cache...")
    processed_dataset.save_to_disk(cache_path)
    
    return processed_dataset

def custom_data_collator(features):
    """Collate examples into batches."""
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in features])
    }


class DroppedLlamaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store gate values for each step
        self.gate_logs = []
        self.log_save_steps = 100
        # Create log directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.args.output_dir, "gate_logs.jsonl")
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)        
        # Get loss
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        aux_loss = outputs.aux_loss if hasattr(outputs, "aux_loss") else None
        
        # Log gate values and losses
        log_stats = {
            "step": self.state.global_step,
            "main_loss": loss.item(),
        }

        if aux_loss is not None:
            log_stats["aux_loss"] = aux_loss.item()
            
        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # Stack gate values: [batch_size, num_layers, 1]
            gate_values = torch.stack(outputs.router_logits, dim=1)
            
            # Add gate statistics to log_stats
            log_stats.update({
                "gate_mean": gate_values.mean().item(),
                "gate_std": gate_values.std().item(),
                "total_open_gates": gate_values.sum(dim=1).mean().item(),  # Average number of open gates
            })
            
            # Log to tensorboard if available
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "loss/main": round(log_stats["main_loss"], 4),
                    "loss/aux": round(log_stats["aux_loss"], 4) if aux_loss is not None else 0,
                    "gate/mean": round(log_stats["gate_mean"], 4),
                    "gate/std": round(log_stats["gate_std"], 4),
                    "gate/total_open": round(log_stats["total_open_gates"], 4)
                })
            
        self.gate_logs.append(log_stats)

        # Write to file periodically
        if self.state.global_step % self.log_save_steps == 0:
            self._write_logs_to_file()

        if return_outputs:
            return loss, outputs
        return loss
    
    def _write_logs_to_file(self):
        """Write accumulated logs to file"""
        if dist.is_initialized() and dist.get_rank() != 0:
            return  # Only write logs from main process
            
        try:
            # Write all new logs
            with open(self.log_file, 'a') as f:
                while self.gate_logs:
                    log_entry = self.gate_logs.pop(0)  # Get and remove first entry
                    f.write(json.dumps(log_entry) + '\n')
                f.flush()  # Ensure it's written to disk
        except Exception as e:
            print(f"Warning: Failed to write logs to file: {e}")
            # Keep the logs in memory if we couldn't write them
            pass
    
    def save_gate_logs(self, output_dir: str):
        """Final save of any remaining logs"""
        self._write_logs_to_file()




def train(
    # Model/data params
    model_dir: str,
    dataset_name: str = "openwebtext",
    output_dir: str = "./output",
    #dataset_cache_dir: str = "./dataset/Skylion007",
    preprocessing_cache_dir: str = "./mapped_datasets",
    # Training hyperparams
    per_device_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 20,
    val_size: int = 10000,
    eval_steps: int = 40,
    save_steps: int = 40,
    max_length: int = 1024,
    # Wandb params
    wandb_project: str = "deepffn",
    wandb_run_name: str = "test",
    # Additional params
    seed: int = 42,
    bf16: bool = True,
    num_workers: int = 32,
    resume_from_checkpoint: str = None, 
):
    # Initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
    # Load model and tokenizer
    print(f"Rank {local_rank} / {world_size} : {torch.cuda.mem_get_info()}")
    config = DroppedLlamaConfig.from_pretrained(model_dir)
    model = DroppedLlamaForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        #device_map='cuda'
    )
    model.config.use_cache = False
    
    if world_size > 1 and local_rank != -1:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Important for performance
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Handle checkpoint resuming
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(output_dir, resume_from_checkpoint)
        if os.path.isdir(checkpoint_path):
            print(f"Loading checkpoint from directory: {checkpoint_path}")
            model = DroppedLlamaForCausalLM.from_pretrained(
                checkpoint_path,
                config=config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map='cuda'
            )

    # Prepare dataset splits
    dataset = load_dataset("openwebtext", num_proc=num_workers)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    print(split_dataset)
    split_dataset['val'] = split_dataset.pop('test') 
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["val"]
    '''
    dataset_cache = os.path.join(dataset_cache_dir, dataset_name)
    train_dataset, val_dataset = create_splits(
        #dataset_name=dataset_name,
        #cache_dir=dataset_cache,
        dataset,
        val_size=val_size
    )
    '''    
    # Preprocess datasets with caching
    preprocessing_cache = os.path.join(preprocessing_cache_dir, dataset_name)
    
    def preprocess_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors=None
        )
        return {
            'input_ids': outputs['input_ids'],
            'attention_mask': outputs['attention_mask'],
            'labels': outputs['input_ids'].copy(),
        }
    
    train_dataset = preprocess_and_cache_dataset(
        dataset=train_dataset,
        cache_dir=preprocessing_cache,
        split_name="train",
        preprocess_fn=preprocess_function,
        num_proc=num_workers
    )

    val_dataset = preprocess_and_cache_dataset(
        dataset=val_dataset,
        cache_dir=preprocessing_cache,
        split_name="val",
        preprocess_fn=preprocess_function,
        num_proc=num_workers
    )

    # Calculate steps
    total_examples = len(train_dataset)
    examples_per_step = per_device_batch_size * gradient_accumulation_steps * world_size
    max_steps = int((total_examples * num_epochs) // examples_per_step)

    # Print training info only from main process
    if local_rank <= 0:  # Main process
        print(f"\n=== Training Configuration ===")
        print(f"World size (num GPUs): {world_size}")
        print(f"Per device batch size: {per_device_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Total batch size per step: {examples_per_step}")
        print(f"Total examples: {total_examples}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Max steps: {max_steps} | not used in trainer")
        print("============================\n")
    
    # Prepare training arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        #max_steps=max_steps,
        num_train_epochs=num_epochs, 
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Add DDP-specific arguments
        local_rank=local_rank,
        ddp_backend="nccl",
        dataloader_pin_memory=True,  # Important for performance
        
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=2,
        
        eval_steps=eval_steps,
        eval_strategy="steps",
        eval_on_start=True,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=2,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        remove_unused_columns=False,
        bf16=bf16,
        dataloader_num_workers=num_workers,
        group_by_length=False,
        
        report_to="wandb" if wandb_project else None,
        run_name=wandb_run_name,
    )

    trainer = DroppedLlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
    )
    

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if local_rank <= 0:
        model.save_pretrained(os.path.join(output_dir, "final_model"))
        trainer.save_gate_logs(output_dir)
    
    return trainer_output

def main():
    parser = argparse.ArgumentParser(description="Train DeepFFN model")
    parser.add_argument("--model-dir", type=str, required=True,
                      help="Directory containing the initialized model")
    parser.add_argument("--dataset-name", type=str, default="roneneldan/TinyStories",
                      help="Dataset name")
    #parser.add_argument("--dataset-config", type=str, default="zyda_crossdeduped-filtered",
    #                  help="Dataset configuration")
    parser.add_argument("--output-dir", type=str, required=True, default="./output",
                      help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                      help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="deepffn",
                      help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, default=None,
                      help="Weights & Biases run name")
    parser.add_argument("--bf16", action="store_true",
                      help="Use bfloat16 precision")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                      help="Resume from checkpoint")
    parser.add_argument("--per_device_batch_size", type=int, default=16,
                      help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Number of gradient accumulation steps")
    args = parser.parse_args()
    
    train(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        bf16=args.bf16,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

if __name__ == "__main__":
    main()