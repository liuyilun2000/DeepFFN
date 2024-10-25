import math
import torch
from torch.nn import Sequential

def convert_trainable_parameters(model, trainable_param_names=None):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += 1
        if trainable_param_names is None:
            param.requires_grad = True
            trainable_params += 1
        else:
            if any(substring in name for substring in trainable_param_names):
                param.requires_grad = True
                trainable_params += 1
            else:
                param.requires_grad = False
    print(
        f"Convert trainable params COUNT: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Print trainable params NUMEL: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )