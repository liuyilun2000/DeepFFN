import os
import sys
import shutil
import zipfile
from pathlib import Path
import kaggle
from tqdm import tqdm

def setup_kaggle_credentials():
    """Verify Kaggle credentials are properly set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_dir.exists():
        print("Creating .kaggle directory...")
        kaggle_dir.mkdir(parents=True)
    
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "kaggle.json not found!\n"
            "1. Go to https://www.kaggle.com/account\n"
            "2. Click 'Create New API Token'\n"
            "3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json"
        )
    
    # Ensure proper permissions
    os.chmod(kaggle_json, 0o600)
    
    # Verify we can access the Kaggle API
    try:
        kaggle.api.authenticate()
        print("Successfully authenticated with Kaggle!")
    except Exception as e:
        raise Exception(f"Failed to authenticate with Kaggle: {e}")

def download_imagenet(output_dir: str, force_download: bool = False):
    """
    Download ImageNet dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        force_download: If True, redownload even if files exist
    """
    # Convert to Path object
    output_dir = Path(output_dir)
    download_dir = output_dir / "download"
    dataset_dir = output_dir / "imagenet"
    
    # Create directories
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Setup Kaggle credentials
    print("Verifying Kaggle credentials...")
    setup_kaggle_credentials()
    
    # Check if already downloaded
    if not force_download and (dataset_dir / "ILSVRC").exists():
        print("ImageNet dataset already exists! Use force_download=True to redownload.")
        return dataset_dir
    
    competition_name = "imagenet-object-localization-challenge"
    zip_path = download_dir / f"{competition_name}.zip"
    
    try:
        # Download competition files
        if not zip_path.exists() or force_download:
            print(f"Downloading ImageNet from Kaggle to {download_dir}...")
            kaggle.api.competition_download_files(
                competition_name,
                path=str(download_dir)
            )
        else:
            print("Found existing download, skipping download step...")
        
        # Extract files
        print(f"Extracting files to {dataset_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc="Extracting"):
                try:
                    zip_ref.extract(member, dataset_dir)
                except Exception as e:
                    print(f"Error extracting {member.filename}: {e}")
        
        # Verify extraction
        required_files = [
            "ILSVRC/Data/CLS-LOC/train",
            "ILSVRC/Data/CLS-LOC/val",
            "LOC_synset_mapping.txt",
            "LOC_val_solution.csv"
        ]
        
        missing = [f for f in required_files 
                  if not (dataset_dir / f).exists()]
        
        if missing:
            raise FileNotFoundError(
                f"Extraction incomplete! Missing files: {missing}"
            )
        
        print(f"Successfully downloaded and extracted ImageNet to {dataset_dir}")
        
        # Clean up download directory
        if zip_path.exists():
            print("Cleaning up downloaded zip file...")
            os.remove(zip_path)
        
        return dataset_dir
        
    except Exception as e:
        print(f"Error downloading/extracting ImageNet: {e}")
        # Clean up on failure
        if zip_path.exists():
            os.remove(zip_path)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download ImageNet from Kaggle")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save the dataset")
    parser.add_argument("--force", action="store_true",
                      help="Force redownload even if files exist")
    
    args = parser.parse_args()
    
    try:
        dataset_path = download_imagenet(args.output_dir, args.force)
        print(f"ImageNet dataset is ready at: {dataset_path}")
    except Exception as e:
        print(f"Failed to download ImageNet: {e}")
        sys.exit(1)