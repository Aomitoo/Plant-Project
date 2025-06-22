from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
import torch

def validate_image(img_tensor, expected_size=(256, 256), normalized=True):
    """
    Проверяет одно изображение на наличие проблем.
    
    Args:
        img_tensor: PyTorch тензор изображения.
        expected_size: Ожидаемый размер изображения (высота, ширина).
        normalized: True, если изображение нормализовано в [0, 1].
    
    Returns:
        bool: True, если изображение валидно, False в противном случае.
    """
    try:
        # Проверка на NaN или inf значения
        if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
            print(f"Обнаружены NaN или inf значения в изображении!")
            return False

        # Проверка размеров
        if img_tensor.shape[1:] != expected_size:
            print(f"Неверные размеры изображения! Ожидалось {expected_size}, получено {img_tensor.shape[1:]}")
            return False

        # Проверка диапазона пиксельных значений
        if normalized:
            if (img_tensor < 0).any() or (img_tensor > 1).any():
                print(f"Пиксельные значения вне диапазона [0, 1] для нормализованного изображения!")
                return False
        else:
            if (img_tensor < 0).any() or (img_tensor > 255).any():
                print(f"Пиксельные значения вне диапазона [0, 255] для ненормализованного изображения!")
                return False

        return True

    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        return False

def validate_dataset(dataset_path, expected_size=(256, 256), normalized=True):
    """
    Проверяет все изображения в датасете.
    
    Args:
        dataset_path: Путь к директории датасета.
        expected_size: Ожидаемый размер изображений.
        normalized: True, если изображения нормализованы.
    
    Returns:
        tuple: (valid_count, total_count, invalid_images)
    """
    # Трансформация для загрузки изображений
    transform = transforms.Compose([
        transforms.Resize(expected_size),
        transforms.ToTensor()
    ])

    # Загрузка датасета
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    total_count = len(dataset)
    valid_count = 0
    invalid_images = []

    for idx, (img, _) in enumerate(dataset):
        if validate_image(img, expected_size, normalized):
            valid_count += 1
        else:
            invalid_images.append((idx, dataset.imgs[idx][0]))

    return valid_count, total_count, invalid_images

if __name__ == "__main__":
    dataset_path = Path("D:/DISUES PLANT/plant_diseases")
    valid_count, total_count, invalid_images = validate_dataset(dataset_path, expected_size=(256, 256), normalized=True)

    print(f"Всего изображений: {total_count}")
    print(f"Валидных изображений: {valid_count}")
    print(f"Невалидных изображений: {len(invalid_images)}")
    if invalid_images:
        print("Детали невалидных изображений:")
        for idx, filepath in invalid_images:
            print(f"Индекс: {idx}, Путь: {filepath}")