import torch
from datasets import load_dataset, Value
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# Load dataset with minimal features
dataset = load_dataset(
    "Zyphra/Zyda-2",
    name="zyda_crossdeduped-filtered",
    split="train",
    streaming=True,
)

# Create a dataloader with multiple workers
dataloader = DataLoader(
    dataset,
    batch_size=1000,
    num_workers=32
)

# Count total examples
total = 0
for batch in tqdm(dataloader, desc="Counting examples"):
    total += len(batch)
    
print(f"\nTotal examples in dataset: {total:,}")