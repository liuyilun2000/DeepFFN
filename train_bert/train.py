import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import copy
import json
import math
import re
import sys
from os.path import join
from pathlib import Path
from typing import List, Optional, Union
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
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast
)

from DeepFFNRoBERTa.configuration_roberta import DeepFFNRoBERTaConfig
from DeepFFNRoBERTa.modelling_roberta import (
    DeepFFNRobertaModel,
    DeepFFNRobertaForSequenceClassification,
    DeepFFNRobertaForMaskedLM,
)

AutoConfig.register("deepffn-roberta", DeepFFNRoBERTaConfig)
AutoModel.register(DeepFFNRoBERTaConfig, DeepFFNRobertaModel)
AutoModelForSequenceClassification.register(DeepFFNRoBERTaConfig, DeepFFNRobertaForSequenceClassification)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def create_splits(dataset_name: str, cache_dir: str, val_size: int = 80000):
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

def train(
    # Model/data params
    model_dir: str,
    dataset_name: str = "openwebtext",
    output_dir: str = "output",
    dataset_cache_dir: str = "../dataset/Skylion007",
    preprocessing_cache_dir: str = "../mapped_datasets/roberta",
    # Training hyperparams
    per_device_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 20,
    val_size: int = 80000,
    eval_steps: int = 500,
    save_steps: int = 500,
    max_length: int = 512,
    # Wandb params
    wandb_project: str = "deepffn",
    wandb_run_name: str = "roberta",
    # Additional params
    seed: int = 42,
    bf16: bool = True,
    num_workers: int = 16,
    resume_from_checkpoint: str = None, 
):
    # Load model and tokenizer
    config = DeepFFNRoBERTaConfig.from_pretrained(model_dir)
    print(f"Model config: vocab_size={config.vocab_size}, "
          f"hidden_size={config.hidden_size}, "
          f"max_position_embeddings={config.max_position_embeddings}")
    model = DeepFFNRobertaForMaskedLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map='cuda'
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, do_lower_case=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    # Handle checkpoint resuming
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(output_dir, resume_from_checkpoint)
        if os.path.isdir(checkpoint_path):
            print(f"Loading checkpoint from directory: {checkpoint_path}")
            model = DeepFFNRobertaForSequenceClassification.from_pretrained(
                checkpoint_path,
                config=config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map='cuda'
            )

    # Prepare dataset splits
    dataset_cache = os.path.join(dataset_cache_dir, dataset_name)
    train_dataset, val_dataset = create_splits(
        dataset_name=dataset_name,
        cache_dir=dataset_cache,
        val_size=val_size
    )
    
    # Preprocess datasets with caching
    preprocessing_cache = os.path.join(preprocessing_cache_dir, dataset_name)
    
    # def preprocess_function(examples):
    #     return tokenizer.encode_plus(
    #         examples['text'],
    #         truncation=True,
    #         max_length=max_length,
    #         padding=True,
    #         return_special_tokens_mask=True
    #     )

    def preprocess_function(examples):
        encodings = [tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=max_length,
            padding=True, 
            return_special_tokens_mask=True
        ) for text in examples['text']]
        
        return {
            'input_ids': [e['input_ids'] for e in encodings],
            'attention_mask': [e['attention_mask'] for e in encodings],
            'special_tokens_mask': [e['special_tokens_mask'] for e in encodings]
        }

    # Use masked lm collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
        
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
    
    total_examples = len(train_dataset)
    examples_per_step = per_device_batch_size * gradient_accumulation_steps
    max_steps = int((total_examples * num_epochs) // examples_per_step)
    
    print("\n=== Dataset and Training Steps Information ===")
    print(f"Total training examples: {total_examples}")
    print(f"Examples per step: {examples_per_step}")
    print(f"Validation set size: {val_size}")
    print(f"Max steps: {max_steps}")
    print("==========================================\n")

    # Prepare training arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        
        eval_steps=eval_steps,
        evaluation_strategy="steps",
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
        save_safetensors=False
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return trainer_output

def main():
    parser = argparse.ArgumentParser(description="Train DeepFFN model")
    parser.add_argument("--model-dir", type=str, required=True,
                      help="Directory containing the initialized model")
    parser.add_argument("--dataset-name", type=str, default="openwebtext",
                      help="Dataset name")
    #parser.add_argument("--dataset-config", type=str, default="zyda_crossdeduped-filtered",
    #                  help="Dataset configuration")
    parser.add_argument("--output-dir", type=str, required=True, default="./output",
                      help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1024,
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
    parser.add_argument("--per-device-batch-size", type=int, default=40,
                      help="Per device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
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