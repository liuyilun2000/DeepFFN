base_model = "FacebookAI/roberta-base"

import argparse
import os
from pathlib import Path
import torch
from typing import Optional

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaTokenizerFast,
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

from utils import *


def initialize_model(
    hidden_size: int,
    intermediate_size_ratio: float,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_mlp_layers: int,
    output_dir: str,
    bf16: bool = True,
    hf_token: Optional[str] = None,
    calc_non_emb_params: bool = False
):
    """Initialize and save a new model with specified configuration."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token,
        max_len=512,
    )
    
    # Create model configuration
    config = DeepFFNRoBERTaConfig(
        max_position_embedding = 514,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size * intermediate_size_ratio),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        num_mlp_layers=num_mlp_layers
    )
    
    # Initialize model
    model = DeepFFNRobertaForMaskedLM(config)

    if bf16:
        model = model.to(torch.bfloat16)
        print("Model initialized as bfloat16")
    
    # Handle parameter conversion
    if calc_non_emb_params:
        convert_trainable_parameters(
            model,
            frozen_param_names=['embed_tokens', 'lm_head', 'embed_in', 'embed_out']
        )
        print_trainable_parameters(model)
    
    convert_trainable_parameters(model)
    print_trainable_parameters(model)
    
    # Save model and configuration
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer successfully saved at {output_dir}")
    return model, tokenizer



def main():
    parser = argparse.ArgumentParser(description="Initialize a DeepFFN-LLaMA model")
    parser.add_argument("--hidden-size", type=int, default=768,
                      help="Hidden size of the model")
    parser.add_argument("--intermediate-size-ratio", type=float, default=4.0,
                      help="Ratio for intermediate size calculation")
    parser.add_argument("--num-hidden-layers", type=int, default=12,
                      help="Number of hidden layers")
    parser.add_argument("--num-attention-heads", type=int, default=12,
                      help="Number of attention heads")
    parser.add_argument("--num-mlp-layers", type=int, default=1,
                      help="Number of MLP layers")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save the initialized model")
    parser.add_argument("--bf16", action="store_true", default=True,
                      help="Convert model to bfloat16 precision")
    parser.add_argument("--hf-token", type=str, default=None,
                      help="HuggingFace token for accessing models")
    parser.add_argument("--calc-non-emb-params", action="store_true", default=True,
                      help="Calculate non-embedding parameters only")
    
    args = parser.parse_args()
    
    initialize_model(
        hidden_size=args.hidden_size,
        intermediate_size_ratio=args.intermediate_size_ratio,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_mlp_layers=args.num_mlp_layers,
        output_dir=args.output_dir,
        bf16=args.bf16,
        hf_token=args.hf_token,
        calc_non_emb_params=args.calc_non_emb_params
    )

if __name__ == "__main__":
    main()
