import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image

class Config:
    DATA_DIR = Path("D:/DISUES PLANT/plant_diseases")
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 0.01
    NUM_CLASSES = 37
    IMG_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_DIM = 1280
    SCALE = 32
    MARGIN = 0.4
    TRAIN_RATIO = 0.7

class DataProcessor:
    def __init__(self):
        self.mean = [0.4432, 0.4937, 0.3295]
        self.std = [0.1955, 0.1907, 0.1878]
        # self.mean = [0.4467, 0.4889, 0.3267]
        # self.std = [0.2299, 0.2224, 0.2289]
        
        self.vall_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.train_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomRotation(60),  # Increase rotation range to ±60 degrees
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Add vertical flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Increase variation
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.5, 1.0)),  # Increase crop variability
            transforms.RandomAffine(degrees=0, shear=10),  # Add shear
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def get_loaders(self):
        full_dataset = datasets.ImageFolder(
            Config.DATA_DIR,
            transform=self.vall_transform
        )
        
        generator = torch.Generator().manual_seed(42)
        total_size = len(full_dataset)
        
        class_counts = torch.zeros(Config.NUM_CLASSES)
        for _, label in full_dataset:
            class_counts[label] += 1
        print("Class distribution:", class_counts)

        # Define 5-fold cross-validation
        k_folds = 5
        fold_size = total_size // k_folds
        indices = torch.arange(total_size)  # Create a tensor of indices [0, 1, ..., total_size-1]
        
        # Generate a random permutation of indices using torch.randperm
        permuted_indices = torch.randperm(total_size, generator=generator)
        
        for fold in range(k_folds):
            # Define indices for validation and training
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k_folds - 1 else total_size
            val_indices = permuted_indices[val_start:val_end]
            train_indices = torch.cat((permuted_indices[:val_start], permuted_indices[val_end:]))

            # Create Subsets for training and validation
            train_subset = torch.utils.data.Subset(full_dataset, train_indices)
            val_subset = torch.utils.data.Subset(full_dataset, val_indices)

            # Apply train_transform to training data
            train_subset.dataset.transform = self.train_transform

            # Create DataLoaders
            train_loader = DataLoader(
                train_subset, batch_size=Config.BATCH_SIZE,
                shuffle=True, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=Config.BATCH_SIZE,
                num_workers=4, pin_memory=True
            )

            yield train_loader, val_loader

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.W)
        self.margin = Config.MARGIN 
        self.scale = Config.SCALE
        
    def forward(self, embeddings, labels):
        # Нормализация весов и эмбеддингов
        W_norm = F.normalize(self.W, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Вычисление косинусов углов
        cos_theta = x_norm @ W_norm.T
        
        # Ограничение значений для стабильности вычислений
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-4, 1.0 - 1e-4)
        
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
        self.margin = Config.MARGIN
        self.scale = Config.SCALE
        
    def forward(self, embeddings, labels):
        # Нормализация весов и эмбеддингов
        W_norm = F.normalize(self.W, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Косинусное расстояние
        logits = x_norm @ W_norm.T
        
        # Ограничение для численной стабильности
        logits = torch.clamp(logits, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Вычитание маргинала для целевых классов
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = logits - self.margin * one_hot
        
        # Масштабирование
        logits = self.scale * logits
        
        return logits

class DiseaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        self.head = ArcFace(Config.FEATURE_DIM, Config.NUM_CLASSES)  
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Added dropout
            nn.Linear(Config.FEATURE_DIM, Config.NUM_CLASSES)
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        
        if labels is not None:
            return self.head(embeddings, labels)
        return self.classifier(embeddings)