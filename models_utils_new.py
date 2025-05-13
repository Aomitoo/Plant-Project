import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image


class Config:
    DATA_DIR = Path("D:/DISUES PLANT/DoctorP_dataset")
    BATCH_SIZE = 32
    NUM_EPOCHS = 100  # Увеличение эпох
    LR = 0.0001       # Снижение learning rate
    NUM_CLASSES = 68
    IMG_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_DIM = 1280  # Размерность эмбеддингов
    SCALE = 32         # Параметры CosFace
    MARGIN = 0.4
    TRAIN_RATIO = 0.8  # Соотношение 80/20


class DataProcessor:
    def __init__(self):
        # Параметры нормализации из исследования
        self.mean = [0.4467, 0.4889, 0.3267]
        self.std = [0.2299, 0.2224, 0.2289]
        
        self.test_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.train_transform = transforms.Compose([
            # transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            # transforms.RandomRotation(45),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            # transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def get_loaders(self):
        # Загрузка всего датасета
        full_dataset = datasets.ImageFolder(
            Config.DATA_DIR,
            transform=self.test_transform  # Базовые преобразования для всего датасета
        )
        
        # Ручное перемешивание перед разделением
        generator = torch.Generator().manual_seed(42)  # Для воспроизводимости
        
        # Разделение 80/20 с сохранением баланса классов
        train_size = int(Config.TRAIN_RATIO * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=generator
        )
        
        # Применяем аугментации только к тренировочному набору
        train_dataset.dataset.transform = self.train_transform
        
        return (
            DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                      shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                      num_workers=4, pin_memory=True)
        )


class CosFace(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.W)
        
    def forward(self, embeddings, labels):
        # Нормализация весов и эмбеддингов
        W_norm = F.normalize(self.W)
        x_norm = F.normalize(embeddings)
        
        # Вычисление косинусной меры
        logits = x_norm @ W_norm.T
        
        # Добавление маргинала только для правильных классов
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1,1), 1)
        logits = logits - one_hot * Config.MARGIN
        
        return Config.SCALE * logits


class DiseaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnext50_32x4d(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Удаление классификатора
        
        # Дополнительные слои для эмбеддингов
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, Config.FEATURE_DIM),
            nn.BatchNorm1d(Config.FEATURE_DIM),
            nn.ReLU()
        )
        
        self.head = CosFace(Config.FEATURE_DIM, Config.NUM_CLASSES)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        
        if labels is not None:
            return self.head(embeddings, labels)
        return embeddings