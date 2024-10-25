import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# First load dataset and convert to iterable with sharding
dataset = load_dataset(
    "Zyphra/Zyda-2",
    name="zyda_crossdeduped-filtered",
    split="train",
    streaming=True
)

# Create a dataloader with multiple workers
dataloader = DataLoader(
    dataset,
    batch_size=1000,  # Large batch size for faster counting
    num_workers=32,    # Use 4 workers
    collate_fn=lambda x: x  # Identity function as collate_fn
)

# Count total examples
total = 0
for batch in tqdm(dataloader, desc="Counting examples"):
    total += len(batch)


print(f"\nTotal examples in dataset: {total:,}")
