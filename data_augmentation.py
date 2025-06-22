import albumentations as A
from PIL import Image
import numpy as np
import os
import shutil

# Define an enhanced augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=45, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0, hue=0,  p=0.8),
    A.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), p=1.0),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.5),
    A.CoarseDropout(max_holes=2, max_height=6, max_width=6, p=0.2)
])

DATA_DIR = "plant_diseases/"
AUG_DIR = "plant_diseases_augmented/"

class_counts = {class_name: len(os.listdir(os.path.join(DATA_DIR, class_name))) for class_name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, class_name))}

for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_name)
    aug_class_dir = os.path.join(AUG_DIR, class_name)
    os.makedirs(aug_class_dir, exist_ok=True)
    if os.path.isdir(class_dir):
        num_images = class_counts[class_name]
        print(f"Class: {class_name}, Original images: {num_images}")
        augment_count = 20 if num_images < 50 else 10
        target_count = min(100, num_images * augment_count)  # Ограничение до 200 изображений
        current_aug_count = 0
        for img_name in os.listdir(class_dir):
            if current_aug_count >= target_count:
                break
            img_path = os.path.join(class_dir, img_name)
            image = np.array(Image.open(img_path))
            # Generate augmented versions and save to the new directory
            for i in range(augment_count):
                if current_aug_count >= target_count:
                    break
                augmented = transform(image=image)
                augmented_image = Image.fromarray(augmented['image'])
                augmented_image.save(os.path.join(aug_class_dir, f"aug_{i}_{img_name}"))
                current_aug_count += 1
        print(f"Class: {class_name}, Total images after augmentation: {len(os.listdir(aug_class_dir))}")