import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image


class Config:
    DATA_DIR = Path("data")
    NEW_DATA_DIR = Path("new_data")
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LR = 0.001
    NUM_CLASSES = 87
    IMG_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("models/checkpoints")  # Папка для чекпоинтов
    BEST_MODEL_PATH = Path("models/best_model.pth")  # Лучшая модель
    METADATA_PATH = Path("models/classes.txt")

    @staticmethod
    def update_num_classes(new_num_classes):
        Config.NUM_CLASSES = new_num_classes


class DataProcessor:
    def __init__(self):
        # Общие трансформы для всех данных
        self.base_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        # Загрузка объединённых классов
        self.full_dataset = self._load_combined_dataset()
        
        # Аугментации только для тренировочных данных
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            self.base_transform
        ])

    def save_classes(self, classes):
        """Сохраняет список классов в файл"""
        Config.METADATA_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(Config.METADATA_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(classes))

    def load_classes(self):
        """Загружает список классов из файла"""
        if Config.METADATA_PATH.exists():
            with open(Config.METADATA_PATH, 'r', encoding='utf-8') as f:
                return f.read().splitlines()
        return []
    
    def _load_combined_dataset(self):
        """Загружает объединённые классы из старого и нового датасетов"""
        new_dataset = datasets.ImageFolder(Config.NEW_DATA_DIR)
        
        # Объединение классов
        all_classes = sorted(list(set(new_dataset.classes)))
        return all_classes

    def get_all_classes(self):
        return self.full_dataset

    def get_loaders(self, use_new_data=True):
        """Загрузка данных с автоматическим разделением"""
        if use_new_data:
            # Загрузка нового датасета из одной папки
            full_dataset = datasets.ImageFolder(
                Config.NEW_DATA_DIR,
                transform=self.base_transform
            )
            
            # Разделение данных
            train_size = int(0.7 * len(full_dataset))
            val_size = int(0.2 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Применяем аугментации к тренировочным данным
            train_dataset.dataset.transform = self.train_transform
            
        else:
            # Старый вариант загрузки
            train_dataset = datasets.ImageFolder(
                Config.DATA_DIR / "train",
                transform=self.train_transform
            )
            val_dataset = datasets.ImageFolder(
                Config.DATA_DIR / "valid",
                transform=self.base_transform
            )
            test_dataset = None

        # Обновляем количество классов
        Config.update_num_classes(len(full_dataset.classes))

        return (
            DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True),
            DataLoader(val_dataset, batch_size=Config.BATCH_SIZE),
            DataLoader(test_dataset, batch_size=Config.BATCH_SIZE) if test_dataset else None
        )


class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):  # Добавляем параметр num_classes
        super().__init__()
        # Используем современный API для загрузки весов
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)  # Используем параметр

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