base_model = "meta-llama/Llama-2-7B-hf"

import argparse
import os
from pathlib import Path
from git import List
import torch
from typing import Optional

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from DroppedLLaMA.configuration_llama import DroppedLlamaConfig
from DroppedLLaMA.modeling_llama import (
    DroppedLlamaModel,
    DroppedLlamaForCausalLM
)

AutoConfig.register("dropped-llama", DroppedLlamaConfig)
AutoModel.register(DroppedLlamaConfig, DroppedLlamaModel)
AutoModelForCausalLM.register(DroppedLlamaConfig, DroppedLlamaForCausalLM)

from utils import *

def initialize_model(
    output_dir: str,
    bf16: bool = True,
    hf_token: Optional[str] = None,
    calc_non_emb_params: bool = False,
    **kwargs
):
    """Initialize and save a new model with specified configuration."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token
    )
    
    # Create model configuration
    config = DroppedLlamaConfig(
        **kwargs
    )
    
    # Initialize model
    model = DroppedLlamaForCausalLM(config)

    if bf16:
        model = model.to(torch.bfloat16)
        print("Model initialized as bfloat16")
    
    # Handle parameter conversion
    if calc_non_emb_params:
        print(f"Non-embedding Parameters:")
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
    ###
    parser.add_argument("--attention-gate", action="store_true", default=True,
                      help="Whether to use attention gating mechanism")
    parser.add_argument("--attention-gate-target", type=int, default=6,
                      help="Target number of attention layers to keep active")
    parser.add_argument("--attention-layers", type=str, default="all",
                  help="Comma-separated list of layer indices to keep attention modules (e.g., '0,1,4,7,10,11') or 'all' for all layers")
    ###
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save the initialized model")
    parser.add_argument("--bf16", action="store_true", default=True,
                      help="Convert model to bfloat16 precision")
    parser.add_argument("--hf-token", type=str, default=None,
                      help="HuggingFace token for accessing models")
    parser.add_argument("--calc-non-emb-params", action="store_true", default=True,
                      help="Calculate non-embedding parameters only")
    
    args = parser.parse_args()
    
    if args.attention_layers.lower() == "all":
            args.attention_layers = list(range(args.num_hidden_layers))
    else:
        try:
            args.attention_layers = [int(x.strip()) for x in args.attention_layers.split(",")]
            if not all(0 <= x < args.num_hidden_layers for x in args.attention_layers):
                raise ValueError(f"All attention layer indices must be between 0 and {args.num_hidden_layers-1}")
        except ValueError as e:
            raise ValueError(f"Invalid attention layers format. Use comma-separated integers or 'all'. Error: {e}")


    if args.attention_gate_target > args.num_hidden_layers:
        raise ValueError(f"attention_gate_target ({args.attention_gate_target}) cannot be larger than "
                        f"num_hidden_layers ({args.num_hidden_layers})")
    
    intermediate_size = int(args.hidden_size * args.intermediate_size_ratio)
    initialize_model(
        output_dir=args.output_dir,
        bf16=args.bf16,
        hf_token=args.hf_token,
        calc_non_emb_params=args.calc_non_emb_params,
        hidden_size=args.hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        attention_gate=args.attention_gate,
        attention_gate_target=args.attention_gate_target,
        attention_layers=args.attention_layers
    )

if __name__ == "__main__":
    main()
