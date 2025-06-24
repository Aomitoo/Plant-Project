import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.utils
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os
import datetime
import numpy as np
import math

class Config:
    BATCH_SIZE = 16
    NUM_CLASSES = 35
    IMG_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKBONE_FEATURE_DIM = 1536
    EMBEDDING_DIM = 1280
    DROPOUT = 0.2

class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, m=1.0):
        super().__init__()
        self.m = m
        self.scale = 32.0
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=1)
        cos_theta = torch.mm(x_norm, W_norm.t())
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cos_theta * self.scale
        acos_theta = torch.acos(cos_theta)
        m_acos = self.m * acos_theta
        floor_term = torch.floor(m_acos / math.pi).long()
        psi_theta = (-1) ** floor_term * torch.cos(m_acos) - 2 * floor_term
        psi_theta = torch.clamp(psi_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * psi_theta + (1.0 - one_hot) * cos_theta
        return output * self.scale

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, stage='stage1'):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(Config.BACKBONE_FEATURE_DIM, Config.EMBEDDING_DIM)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.stage = stage
        if stage == 'stage1':
            self.head = SphereFace(Config.EMBEDDING_DIM, num_classes, m=1.0)
        else:
            self.head = ClassificationHead(Config.EMBEDDING_DIM, num_classes)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.dropout(x)
        if self.stage == 'stage1':
            return self.head(x, labels)
        return self.head(x)

# Класс для центрированного обрезания до 1:1
class CenterCropSquare:
    def __call__(self, img):
        width, height = img.size
        size = min(width, height)
        return transforms.functional.center_crop(img, size)

def evaluate_test_set(test_dir="test_data/"):
    # Загрузка модели
    model = DiseaseClassifier(num_classes=Config.NUM_CLASSES, stage='stage1')
    state_dict = torch.load('models/Efficient_70%_BEST.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(Config.DEVICE)

    # # Трансформации: центрированное обрезание до 1:1, затем ресайз до 128x128
    # transform = transforms.Compose([
    #     CenterCropSquare(),  # Обрезка до квадрата (1:1)
    #     transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),  # Ресайз до 128x128
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4425, 0.4931, 0.3288], std=[0.1961, 0.1912, 0.1884])
    # ])

    transform = transforms.Compose([
        # CenterCropSquare(),  # Обрезка до квадрата (1:1)
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),  # Ресайз до 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4569, 0.5046, 0.3590], std=[0.2169, 0.2138, 0.2120])])

    # Загрузка тестового набора
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    class_names = test_dataset.classes

    # Оценка с обработкой ошибок и сохранением батчей
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    batch_count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            try:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Сохранение 1-го (0), 5-го (4) и 8-го (7) батчей
                if batch_count in [0, 4, 7]:
                    batch_dir = setup_metrics_dir()
                    torch.save({'inputs': inputs.cpu(), 'labels': labels.cpu()}, f"{batch_dir}/batch_{batch_count}.pt")
                    # Сохранение изображений как сетку
                    torchvision.utils.save_image(inputs.cpu(), f"{batch_dir}/batch_{batch_count}.png", nrow=int(inputs.size(0) ** 0.5), normalize=True)

                batch_count += 1
            except Exception as e:
                print(f"Error processing batch: {e}. Skipping problematic data.")

    if total_samples == 0:
        print("No valid images processed. Check your test dataset.")
        return 0.0

    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2%} | Prec: {test_precision:.2%} | Rec: {test_recall:.2%} | F1: {test_f1:.2%}")

    # Графики
    metrics_dir = setup_metrics_dir()
    plot_metrics(test_acc, test_loss, test_precision, test_recall, test_f1, metrics_dir)
    plot_confusion_matrix(all_labels, all_preds, class_names, metrics_dir)

    return test_acc

def plot_metrics(test_acc, test_loss, test_precision, test_recall, test_f1, metrics_dir):
    plt.figure(figsize=(12, 6))
    
    # График точности
    plt.subplot(1, 2, 1)
    plt.bar(['Test Accuracy'], [test_acc], color='green')
    plt.title('Test Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    
    # График потерь
    plt.subplot(1, 2, 2)
    plt.bar(['Test Loss'], [test_loss], color='red')
    plt.title('Test Loss')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/test_metrics.png")
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, class_names, metrics_dir):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/test_confusion_matrix.png")
    plt.close()

def setup_metrics_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f"metrics/{timestamp}_test"
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    evaluate_test_set()