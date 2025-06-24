import os
import shutil
from pathlib import Path
from torchvision import datasets
from torch.utils.data import random_split
import torch

def split_dataset_to_folders(data_dir, train_ratio=0.8):
    """
    Split dataset in data_dir into train and val folders using random_split with seed 42.
    Creates plant_diseases/train/class_name and plant_diseases/val/class_name directories.
    
    Args:
        data_dir (str): Path to dataset directory (e.g., 'plant_diseases/')
        train_ratio (float): Proportion of data for training (default: 0.8)
    """
    # Ensure data_dir exists
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(data_dir)
    classes, _ = full_dataset.find_classes(data_dir)
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    # Perform random split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Create train and val directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Create class subdirectories
    for class_name in classes:
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
    
    # Move files to train folder
    for idx in train_dataset.indices:
        img_path, label = full_dataset.imgs[idx]
        class_name = classes[label]
        dest_path = train_dir / class_name / Path(img_path).name
        shutil.move(img_path, dest_path)
    
    # Move files to val folder
    for idx in val_dataset.indices:
        img_path, label = full_dataset.imgs[idx]
        class_name = classes[label]
        dest_path = val_dir / class_name / Path(img_path).name
        shutil.move(img_path, dest_path)
    
    # Remove empty original class directories
    for class_name in classes:
        class_dir = data_dir / class_name
        if class_dir.exists() and not any(class_dir.iterdir()):
            class_dir.rmdir()
    
    print(f"Dataset split completed: {train_size} images in train, {val_size} images in val.")

if __name__ == "__main__":
    data_dir = "plant_diseases/"
    split_dataset_to_folders(data_dir, train_ratio=0.8)