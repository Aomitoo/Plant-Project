import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from PIL import Image
import numpy as np
from collections import Counter
import datetime
from torchvision.datasets import ImageFolder

def analyze_datasets(train_val_dir="plant_diseases", test_dir="test_data/"):
    # Combine train and val datasets
    train_dir = os.path.join(train_val_dir, "train")
    val_dir = os.path.join(train_val_dir, "val")
    combined_samples = []
    combined_targets = []
    
    # Load train dataset
    train_dataset = ImageFolder(train_dir)
    combined_samples.extend(train_dataset.samples)
    combined_targets.extend(train_dataset.targets)
    
    # Load val dataset and adjust target indices if necessary
    val_dataset = ImageFolder(val_dir)
    train_classes = train_dataset.classes
    val_classes = val_dataset.classes
    
    # Ensure class consistency
    if set(train_classes) != set(val_classes):
        print("Warning: Train and validation classes do not match!")
        print(f"Train classes: {train_classes}")
        print(f"Validation classes: {val_classes}")
    
    # Map val classes to train class indices
    class_to_idx = {cls: idx for idx, cls in enumerate(train_classes)}
    val_samples_adjusted = [(path, class_to_idx[val_dataset.classes[target]]) for path, target in val_dataset.samples]
    combined_samples.extend(val_samples_adjusted)
    combined_targets.extend([class_to_idx[val_dataset.classes[target]] for target in val_dataset.targets])
    
    # Create a combined dataset
    class CombinedDataset:
        def __init__(self, samples, targets, classes):
            self.samples = samples
            self.targets = targets
            self.classes = classes

    combined_dataset = CombinedDataset(combined_samples, combined_targets, train_classes)
    
    # Load test dataset
    test_dataset = ImageFolder(test_dir)

    # Get class names and counts
    combined_classes = combined_dataset.classes
    test_classes = test_dataset.classes
    combined_counts = Counter(combined_dataset.targets)
    test_counts = Counter(test_dataset.targets)

    # Get image dimensions (sample one image per class)
    combined_dims = {}
    test_dims = {}
    for cls in combined_classes:
        cls_idx = combined_classes.index(cls)
        img_path = [p for p, t in combined_dataset.samples if t == cls_idx][0]
        with Image.open(img_path) as img:
            combined_dims[cls] = img.size
    for cls in test_classes:
        cls_idx = test_classes.index(cls)
        img_path = [p for p, t in test_dataset.samples if t == cls_idx][0]
        with Image.open(img_path) as img:
            test_dims[cls] = img.size

    # Calculate statistics
    combined_total = sum(combined_counts.values())
    test_total = sum(test_counts.values())
    combined_class_dist = {combined_classes[k]: v / combined_total for k, v in combined_counts.items()}
    test_class_dist = {test_classes[k]: v / test_total for k, v in test_counts.items()}
    combined_balance = np.std(list(combined_class_dist.values())) / np.mean(list(combined_class_dist.values()))
    test_balance = np.std(list(test_class_dist.values())) / np.mean(list(test_class_dist.values()))

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"metrics/{timestamp}_dataset_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Save class distribution plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(combined_class_dist.keys(), combined_class_dist.values())
    plt.title("Combined Train+Val Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Proportion")
    plt.xticks(rotation=90)
    plt.subplot(1, 2, 2)
    plt.bar(test_class_dist.keys(), test_class_dist.values())
    plt.title("Test Set Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Proportion")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.close()

    # Print information
    print("=== Dataset Analysis ===")
    print("\nCombined Train+Validation Dataset:")
    print(f"Total Images: {combined_total}")
    print(f"Number of Classes: {len(combined_classes)}")
    print(f"Classes: {combined_classes}")
    print("Images per Class:")
    for cls, count in combined_counts.items():
        print(f"  {combined_classes[cls]}: {count} images")
    print(f"Class Balance (std/mean): {combined_balance:.4f}")
    print("Sample Image Dimensions:")
    for cls, dim in combined_dims.items():
        print(f"  {cls}: {dim[0]}x{dim[1]}")

    print("\nTest Dataset:")
    print(f"Total Images: {test_total}")
    print(f"Number of Classes: {len(test_classes)}")
    print(f"Classes: {test_classes}")
    print("Images per Class:")
    for cls, count in test_counts.items():
        print(f"  {test_classes[cls]}: {count} images")
    print(f"Class Balance (std/mean): {test_balance:.4f}")
    print("Sample Image Dimensions:")
    for cls, dim in test_dims.items():
        print(f"  {cls}: {dim[0]}x{dim[1]}")

    print(f"\nClass distribution plot saved to: {output_dir}/class_distribution.png")

if __name__ == "__main__":
    analyze_datasets()