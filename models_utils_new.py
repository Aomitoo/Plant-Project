import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image


class Config:
    DATA_DIR = Path("D:/DISUES PLANT/DoctorP_dataset")
    BATCH_SIZE = 32
    NUM_EPOCHS = 30  # Увеличение эпох
    LR = 0.0001       # Снижение learning rate
    NUM_CLASSES = 68
    IMG_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_DIM = 1280  # Размерность эмбеддингов
    SCALE = 32         # Параметры CosFace
    MARGIN = 0.5
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
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.6, 1.0)),
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

        # проверка баланса классов
        class_counts = torch.zeros(Config.NUM_CLASSES)
        for _, label in full_dataset:
            class_counts[label] += 1
        print("Class distribution:", class_counts)
        
        return (
            DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                      shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                      num_workers=4, pin_memory=True)
        )

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.W)
        self.margin = Config.MARGIN  # Используем параметр из конфига
        self.scale = Config.SCALE
        
    def forward(self, embeddings, labels):
        # Нормализация весов и эмбеддингов
        W_norm = F.normalize(self.W, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Вычисление косинусов углов
        cos_theta = x_norm @ W_norm.T
        
        # Ограничение значений для стабильности вычислений
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Вычисление углов theta
        theta = torch.acos(cos_theta)
        
        # Добавление маржина к углам для целевых классов
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        theta_margin = theta + self.margin * one_hot
        
        # Вычисление нового косинуса с маржином
        cos_theta_margin = torch.cos(theta_margin)
        
        # Комбинирование логитов
        logits = self.scale * (one_hot * cos_theta_margin + (1 - one_hot) * cos_theta)
        
        return logits
    
class CosFace(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.W)
        
    def forward(self, embeddings, labels):
        # Нормализация весов и эмбеддингов
        W_norm = F.normalize(self.W, p=2, dim=1)  # Явно укажите параметры
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Вычисление косинусной меры
        logits = x_norm @ W_norm.T
        
        # Исправьте вычитание маргинала:
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1,1), Config.MARGIN)  # Маргинал добавляется только к target-логитам
        logits = logits - one_hot
        
        return Config.SCALE * logits


class DiseaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Удаление классификатора
        
        # Заморозка весов backbone
        for param in self.backbone.parameters():
            param.requires_grad = False 

        # Добавьте адаптивный пулинг
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Важно для любого размера!
        self.backbone.fc = nn.Identity()
        
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, Config.FEATURE_DIM),
            nn.BatchNorm1d(Config.FEATURE_DIM),
            nn.ReLU()
        )
        # self.head = CosFace(Config.FEATURE_DIM, Config.NUM_CLASSES)  # Старая версия
        self.head = ArcFace(Config.FEATURE_DIM, Config.NUM_CLASSES)     # Новая версия

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        
        if labels is not None:
            return self.head(embeddings, labels)
        return embeddings