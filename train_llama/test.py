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


def create_splits(dataset_name: str, dataset_config: str, cache_dir: str, val_size: int = 1000):
    """Create training and validation splits from streaming dataset."""
    print(f"Loading dataset {dataset_name}: {dataset_config}...")
    full_dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
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

model_dir = "DeepFFNLLaMA/DeepFFNLLaMA_768_12_2-2"
checkpoint_dir = "output/768_12_2-2/checkpoint-620"


model_dir = "DeepFFNLLaMA/DeepFFNLLaMA_768_4_4-3"
checkpoint_dir = "output/768_4_4-3/checkpoint-620"
bf16=True


# Load model and tokenizer
config = DeepFFNLlamaConfig.from_pretrained(checkpoint_dir)
model = DeepFFNLlamaForCausalLM.from_pretrained(
    checkpoint_dir,
    config=config,
    torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token




prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum"

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_new_tokens=80)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]







dataset = load_dataset("openwebtext", trust_remote_code=True, encoding='utf-8')