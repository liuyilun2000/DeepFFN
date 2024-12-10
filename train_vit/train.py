import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import math
from pathlib import Path
from typing import Optional, Dict, List
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import wandb
from tqdm import tqdm
import transformers
from transformers import (
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)

from DeepFFNViT.configuration_vit import DeepFFNViTConfig
from DeepFFNViT.modeling_vit import DeepFFNViTForMaskedImageModeling

class KaggleImageNetDataset(Dataset):
    def __init__(self, root_dir: str, split: str, processor: ViTImageProcessor):
        self.root_dir = Path(root_dir)
        self.split = split
        self.processor = processor
        self.samples = []
        self.patch_size = processor.size['height'] // 16  # For ViT-Base
        self.num_patches = (processor.size['height'] // 16) ** 2
        
        if split == "train":
            data_dir = self.root_dir / "ILSVRC/Data/CLS-LOC/train"
            for class_dir in data_dir.iterdir():
                if class_dir.is_dir():
                    self.samples.extend(list(class_dir.glob("*.JPEG")))
        else:
            data_dir = self.root_dir / "ILSVRC/Data/CLS-LOC/val"
            self.samples.extend(list(data_dir.glob("*.JPEG")))
        
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=processor.image_mean,
                    std=processor.image_std
                )
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=processor.image_mean,
                    std=processor.image_std
                )
            ])
        
        print(f"Loaded {len(self.samples)} images for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def create_random_mask(self):
        """Create random boolean mask for masked image modeling."""
        return torch.randint(0, 2, (self.num_patches,), dtype=torch.bool)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.transforms(image)
            bool_masked_pos = self.create_random_mask()
            return {
                "pixel_values": pixel_values,
                "bool_masked_pos": bool_masked_pos
            }
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def mim_collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle pixel values and boolean masks."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    bool_masked_pos = torch.stack([example["bool_masked_pos"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "bool_masked_pos": bool_masked_pos
    }

def create_dataloaders(
    data_dir: str,
    processor: ViTImageProcessor,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    print("Creating datasets...")
    train_dataset = KaggleImageNetDataset(data_dir, "train", processor)
    val_dataset = KaggleImageNetDataset(data_dir, "val", processor)
    
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=mim_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=mim_collate_fn
    )
    
    return train_loader, val_loader


def compute_metrics(pred):
    """Compute metrics for masked image modeling."""
    # pred.predictions is the reconstructed image
    # pred.label_ids are the original images
    # pred.inputs['bool_masked_pos'] contains the mask

    loss = pred.predictions[0]  # This is the loss returned by the model
    reconstructed_pixels = pred.predictions[1]  # This is the reconstruction
    
    return {
        "mim_loss": loss.mean().item(),  # Overall masked image modeling loss
        "reconstruction_loss": torch.nn.functional.l1_loss(
            pred.label_ids,
            reconstructed_pixels,
            reduction="mean"
        ).item()  # L1 loss between original and reconstructed pixels
    }

def get_device(bf16: bool = False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if bf16 else torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32  # Always use float32 for CPU
        if bf16:
            print("Warning: bfloat16 not supported on CPU. Using float32 instead.")
    return device, dtype

def train(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    per_device_batch_size: int = 32,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 15,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_steps: int = 1000,  # Reduced for shorter training
    max_eval_steps: int = 100,
    wandb_project: str = "deepffn-vit",
    wandb_name: str = None,
    seed: int = 42,
    bf16: bool = True,
    num_workers: int = 16,
    resume_from_checkpoint: str = None,
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    device, dtype = get_device(bf16)
    print(f"Using device: {device}, dtype: {dtype}")

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Initialize wandb
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "model_dir": model_dir,
                "batch_size": per_device_batch_size * gradient_accumulation_steps,
                "eval_batch_size": eval_batch_size,
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "max_eval_steps": max_eval_steps,
            }
        )

    if device.type == "cpu":
        print("Running on CPU - adjusting number of workers")
        num_workers = min(num_workers, cpu_count() // 2)
        print(f"Using {num_workers} workers")
    
    # Load model configuration and processor
    print("Loading model configuration and processor...")
    config = DeepFFNViTConfig.from_pretrained(model_dir)
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        do_resize=True,
        size=224
    )
    
    # Initialize model
    print("Initializing model...")
    model = DeepFFNViTForMaskedImageModeling.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model = model.to(device)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        processor=processor,
        train_batch_size=per_device_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers
    )
    
    # Initialize training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        bf16=bf16,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=200,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="reconstruction_loss",
        greater_is_better=False,  # Lower reconstruction loss is better
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_name,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=device.type == "cuda",
        no_cuda=device.type == "cpu",
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("Starting training...")
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(f"{output_dir}/final_model")
    
    if wandb_project:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train DeepFFN-ViT model")
    parser.add_argument("--model-dir", type=str, required=True,
                      help="Directory containing the initialized model")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory containing ImageNet dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                      help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3,
                      help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="deepffn-vit",
                      help="Weights & Biases project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                      help="Weights & Biases run name")
    parser.add_argument("--bf16", action="store_true",
                      help="Use bfloat16 precision")
    parser.add_argument("--resume", type=str, default=None,
                      help="Resume from checkpoint")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU training")
    
    args = parser.parse_args()
    
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        bf16=args.bf16,
        resume_from_checkpoint=args.resume
    )

if __name__ == "__main__":
    main()