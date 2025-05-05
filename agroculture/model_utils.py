import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image


class Config:
    DATA_DIR = Path("data")
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LR = 0.001
    NUM_CLASSES = 38
    IMG_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataProcessor:
    def __init__(self):
        self.test_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def get_loaders(self):
        train_data = datasets.ImageFolder(
            Config.DATA_DIR / "train",
            transform=self.train_transform
        )
        test_data = datasets.ImageFolder(
            Config.DATA_DIR / "valid",
            transform=self.test_transform
        )
        return (
            DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test_data, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
        )


class DiseaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet34(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        return self.base_model(x)


def predict(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(1).item()
    return class_names[pred_idx]