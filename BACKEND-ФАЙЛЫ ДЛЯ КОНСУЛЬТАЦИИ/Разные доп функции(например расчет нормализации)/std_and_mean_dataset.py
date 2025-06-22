import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

if __name__ == '__main__':
    # Загружаем датасет без нормализации
    dataset = datasets.ImageFolder(
        Path("D:/DISUES PLANT/plant_diseases"),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()]
        ))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    print(f"Mean: {mean}")
    print(f"Std: {std}")