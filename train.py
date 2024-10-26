HF_TOKEN = "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"


import argparse
import copy
import json
import math
import os
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


def create_splits(dataset_name: str, dataset_config: str, cache_dir: str, val_size: int = 10000):
    """Create training and validation splits from streaming dataset."""
    print(f"Loading dataset {dataset_name}: {dataset_config}...")
    full_dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
        cache_dir=cache_dir
    )
    
    val_dataset = full_dataset.take(val_size)
    train_dataset = full_dataset.skip(val_size)
    
    return train_dataset, val_dataset


def count_examples_in_stream(dataset, sample_size=1000):
    """
    Estimate total examples in a streaming dataset by sampling.
    Returns both a sample count and estimated total.
    """
    print("Sampling dataset to estimate size...")
    # Take a small sample and count time
    sample = dataset.take(sample_size)
    sample_count = 0
    for _ in tqdm(sample, total=sample_size):
        sample_count += 1
    
    print(f"Found {sample_count} examples in sample")
    return sample_count



def preprocess_function(examples, tokenizer, max_length: int = 1024):
    """Tokenize and prepare the examples."""
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


def custom_data_collator(features):
    """Collate examples into batches."""
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in features])
    }

def compute_metrics(eval_preds):
    """Compute perplexity and other metrics."""
    logits, labels = eval_preds
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits).view(-1, logits.shape[-1]), 
        torch.tensor(labels).view(-1)
    )
    perplexity = torch.exp(loss)
    
    return {
        "perplexity": perplexity.item()
    }


def train(
    # Model/data params
    model_dir: str,
    dataset_name: str = "Zyphra/Zyda-2",
    dataset_config: str = "zyda_crossdeduped-filtered",
    output_dir: str = "./output",
    # Training hyperparams
    batch_size: int = 16,
    micro_batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    val_set_size: int = 10000,
    use_gradient_checkpointing: bool = False,
    eval_step: int = 200,
    save_step: int = 200,
    max_length: int = 4096,
    # Wandb params
    wandb_project: str = "deepffn",
    wandb_run_name: str = "test",
    # Additional params
    seed: int = 42,
    bf16: bool = True,
    num_workers: int = 4,
):
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

    # Prepare dataset
    cache_dir = f"./dataset/{dataset_config}"
    train_dataset, val_dataset = create_splits(dataset_name, dataset_config, cache_dir, val_set_size)
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    examples_per_step = micro_batch_size * gradient_accumulation_steps
    
    print("\n=== Dataset and Training Steps Information ===")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Examples processed per step: {examples_per_step}")
    print(f"Validation set size: {val_set_size}")
    
    # For streaming datasets, we still need to specify max_steps
    # But now we have better information about the actual dataset size
    total_examples = 248e6  # Use actual count instead of estimate
    max_steps = (total_examples * num_epochs) // examples_per_step
    
    print(f"Estimated total steps: {max_steps} (based on sample count)")
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
        logging_steps=10,      
        optim="adamw_torch", 

        eval_strategy="steps" if val_set_size > 0 else "no",
        eval_steps=eval_step if val_set_size > 0 else None,
        save_strategy="steps",
        save_steps=save_step,
        output_dir=output_dir,
        save_total_limit=3,
        
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
        ),
        compute_metrics=compute_metrics,
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
    
    trainer_output = trainer.train()
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return trainer_output

def main():
    parser = argparse.ArgumentParser(description="Train DeepFFN model")
    parser.add_argument("--model-dir", type=str, required=True,
                      help="Directory containing the initialized model")
    parser.add_argument("--dataset-name", type=str, default="Zyphra/Zyda-2",
                      help="Dataset name")
    parser.add_argument("--dataset-config", type=str, default="zyda_crossdeduped-filtered",
                      help="Dataset configuration")
    parser.add_argument("--output-dir", type=str, required=True, default="./output",
                      help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Total batch size")
    parser.add_argument("--micro-batch-size", type=int, default=16,
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
    
    args = parser.parse_args()
    
    train(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        bf16=args.bf16
    )

if __name__ == "__main__":
    main()