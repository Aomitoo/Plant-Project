import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path

class Config:
    DATA_DIR = Path("D:/DISUES PLANT/plant_diseases")
    BATCH_SIZE = 16  # Уменьшено для экономии памяти
    NUM_EPOCHS = 30  # Уменьшено для более быстрого эксперимента
    LR = 0.0001
    NUM_CLASSES = 37
    IMG_SIZE = 224  # Уменьшено до стандартного размера ConvNeXt
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_DIM = 1280
    SCALE = 32
    MARGIN = 0.5
    TRAIN_RATIO = 0.7

class DataProcessor:
    def __init__(self):
        # self.mean = [0.4467, 0.4889, 0.3267]
        # self.std = [0.2299, 0.2224, 0.2289]
        self.mean = [0.4425, 0.4931, 0.3288]
        self.std = [0.1961, 0.1912, 0.1884]
        
        self.vall_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Убраны аугментации для соответствия исследованию
        self.train_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_loaders(self):
        sample_image, _ = next(iter(datasets.ImageFolder(Config.DATA_DIR, transform=transforms.ToTensor())))
        print(f"Sample image min: {torch.min(sample_image)}, max: {torch.max(sample_image)}")

        full_dataset = datasets.ImageFolder(
            Config.DATA_DIR,
            transform=self.vall_transform
        )
        print(full_dataset.classes)
        
        generator = torch.Generator().manual_seed(42)
        
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        train_dataset.dataset.transform = self.train_transform

        class_counts = torch.zeros(Config.NUM_CLASSES)
        for _, label in full_dataset:
            class_counts[label] += 1
        print("Class distribution:", class_counts)
        
        return (
            DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                    shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                    num_workers=4, pin_memory=True),
            DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                    num_workers=4, pin_memory=True)  
        )

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale=32.0, margin=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.W)
        self.W.data = F.normalize(self.W.data, p=2, dim=1)
        self.margin = margin
        self.scale = scale
        print(f"ArcFace W norm: {torch.norm(self.W)}")
        
    def forward(self, embeddings, labels=None):
        W_norm = F.normalize(self.W, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        cos_theta = x_norm @ W_norm.T
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-4, 1.0 - 1e-4)
        
        if labels is None:
            return self.scale * cos_theta
        
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        theta_margin = theta + self.margin * one_hot
        
        cos_theta_margin = torch.cos(theta_margin)
        logits = self.scale * (one_hot * cos_theta_margin + (1 - one_hot) * cos_theta)
        
        
        return logits

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, scale=32.0, margin=0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        
        phi = cosine - self.margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=1280, head_type='arcface'):
        super(DiseaseClassifier, self).__init__()
        self.backbone = models.convnext_small(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, feature_dim)
        if head_type == 'arcface':
            self.head = ArcFace(feature_dim, num_classes, scale=32.0, margin=0.5)
        elif head_type == 'cosface':
            self.head = CosFace(feature_dim, num_classes, scale=32.0, margin=0.4)
        else:
            raise ValueError("Unsupported head_type. Use 'arcface' or 'cosface'.")

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.head(x, labels)
        return x