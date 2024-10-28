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
from typing import List, Optional, Union

import fire
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from tqdm import tqdm

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from datasets import load_dataset

import transformers
from transformers import (
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

from utils import *


def preprocess_function(examples, tokenizer, max_length: int = 1024):
    """Tokenize and prepare the examples."""
    texts = []
    for text in examples['text']:
        texts.append(text + tokenizer.eos_token)
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    return {
        'input_ids': outputs['input_ids'],
        'attention_mask': outputs['attention_mask'],
        'labels': outputs['input_ids'].copy(),
    }


def custom_data_collator(features):
    """Collate examples into batches."""
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in features])
    }


def train(
    # Model/data params
    model_dir: str,
    dataset_name: str = "roneneldan/TinyStories",
    output_dir: str = "./output",
    # Training hyperparams
    batch_size: int = 16,
    micro_batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 20,
    #val_set_size: int = 2000,
    use_gradient_checkpointing: bool = False,
    eval_step: int = 20,
    save_step: int = 20,
    max_length: int = 1024,
    # Wandb params
    wandb_project: str = "deepffn",
    wandb_run_name: str = "test",
    # Additional params
    seed: int = 42,
    bf16: bool = True,
    num_workers: int = 4,
    resume_from_checkpoint: str = None, 
):
    print(
        f"Training model with params:\n"
        f"=== Model/Data Parameters ===\n"
        f"model_dir: {model_dir}\n"
        f"dataset_name: {dataset_name}\n"
        f"output_dir: {output_dir}\n"
        f"\n=== Training Hyperparameters ===\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"weight_decay: {weight_decay}\n"
        f"warmup_steps: {warmup_steps}\n"
        #f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"eval_step: {eval_step}\n"
        f"save_step: {save_step}\n"
        f"max_length: {max_length}\n"
        f"\n=== Weights & Biases Configuration ===\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"\n=== Additional Parameters ===\n"
        f"seed: {seed}\n"
        f"bf16: {bf16}\n"
        f"num_workers: {num_workers}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )

    # Load model and tokenizer
    config = DeepFFNLlamaConfig.from_pretrained(model_dir)
    model = DeepFFNLlamaForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"### Model successfully loaded at {model_dir} ###")

    
    # Handle checkpoint resuming
    if resume_from_checkpoint:
        resume_from_checkpoint = os.path.join(output_dir, resume_from_checkpoint)
        print(f"##### Attempting to resume from checkpoint: {resume_from_checkpoint} #####")
        
        # Check for HF checkpoint directory
        if os.path.isdir(resume_from_checkpoint):
            print(f"Loading checkpoint from directory: {resume_from_checkpoint}")
            model = DeepFFNLlamaForCausalLM.from_pretrained(
                resume_from_checkpoint,
                config=config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map='auto'
            )
            print(f"##### Successfully loaded checkpoint from {resume_from_checkpoint} #####")
        else:
            print(f"##### Checkpoint directory {resume_from_checkpoint} not found #####")

    # Prepare dataset
    
    train_dataset = load_dataset(
        dataset_name,
        split="train"
    )
    val_dataset = load_dataset(
        dataset_name,
        split="validation"
    )
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    examples_per_step = micro_batch_size * gradient_accumulation_steps
    
    train_set_size = len(train_dataset) 
    val_set_size = len(val_dataset)
    max_steps = int((train_set_size * num_epochs) // examples_per_step)
    
    print("\n=== Dataset and Training Steps Information ===")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Examples processed per step: {examples_per_step}")
    print(f"Validation set size: {val_set_size}")
    print(f"Estimated total steps: {max_steps}")
    print("==========================================\n")

    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Prepare training arguments
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=int(micro_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        max_steps=int(max_steps),
        warmup_steps=int(warmup_steps),
        num_train_epochs=int(num_epochs),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bf16=bf16,
        logging_steps=2,      
        optim="adamw_torch", 

        eval_strategy="steps" if val_set_size > 0 else "no",
        eval_steps=eval_step if val_set_size > 0 else None,
        save_strategy="steps",
        save_steps=save_step,
        output_dir=output_dir,
        save_total_limit=3,

        #resume_from_checkpoint=resume_from_checkpoint,  # Enable checkpoint resuming
        ignore_data_skip=False,  
        
        load_best_model_at_end=True if val_set_size > 0 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        group_by_length=False,
        
        report_to="wandb" if wandb_project else None,
        run_name=wandb_run_name,
    )

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    
    print(f"##### Trainer successfully initialized for {wandb_run_name} #####")
    model.config.use_cache = False

    old_state_dict = model.state_dict

    model.state_dict = (
        lambda self, *_, **__: {
            name: param.data
            for name, param in self.named_parameters()
            if param.requires_grad
        }
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        print(f"##### Compiling for {wandb_run_name} #####")
        model = torch.compile(model)
        print(f"##### Compiling finished #####")
    
    trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    
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
    parser.add_argument("--batch-size", type=int, default=1048,
                      help="Total batch size")
    parser.add_argument("--micro-batch-size", type=int, default=64,
                      help="Micro batch size")
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
    
    args = parser.parse_args()
    
    train(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        #dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        bf16=args.bf16,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

if __name__ == "__main__":
    main()