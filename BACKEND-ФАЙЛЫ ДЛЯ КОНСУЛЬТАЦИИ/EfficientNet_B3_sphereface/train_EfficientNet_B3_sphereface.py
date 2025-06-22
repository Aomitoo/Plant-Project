# Константы путей
DATA_DIR = "plant_diseases/"  # Путь к папке с датасетом
MODEL_SAVE_PATH = "models/efficientnet_sphereface"  # Путь для сохранения модели
METRICS_DIR = "metrics/"  # Путь для метрик и графиков

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os
import datetime
import seaborn as sns
import numpy as np
import math

class Config:
    BATCH_SIZE = 32  # Уменьшено для GTX 1650
    NUM_EPOCHS_STAGE1 = 5
    NUM_EPOCHS_STAGE2 = 10
    LR = 0.026
    NUM_CLASSES = 37
    IMG_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURE_DIM = 1536  # Для EfficientNet_B3
    MARGIN = 4.0  # Для SphereFace
    TRAIN_RATIO = 0.7
    DROPOUT = 0.2
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 10.0  # Увеличено для стабильности

class DataProcessor:
    def __init__(self):
        # Нормализация из исследования
        self.mean = [0.4425, 0.4931, 0.3288]
        self.std = [0.1961, 0.1912, 0.1884]

        self.val_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.train_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_loaders(self):
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=self.val_transform)
        generator = torch.Generator().manual_seed(42)
        total_size = len(full_dataset)
        train_size = int(Config.TRAIN_RATIO * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        train_dataset.dataset.transform = self.train_transform

        return (
            DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=2, pin_memory=True),
            DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=2, pin_memory=True)
        )

class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, m=4.0):
        super().__init__()
        self.m = m
        self.scale = 32.0  # Уменьшено для стабильности
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=1)
        cos_theta = torch.mm(x_norm, W_norm.t())
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cos_theta * self.scale

        # A-Softmax
        acos_theta = torch.acos(cos_theta)
        m_acos = self.m * acos_theta
        floor_term = torch.floor(m_acos / math.pi).long()
        psi_theta = (-1) ** floor_term * torch.cos(m_acos) - 2 * floor_term
        psi_theta = torch.clamp(psi_theta, -1.0 + 1e-7, 1.0 - 1e-7)  # Проверка на NaN/Inf
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * psi_theta + (1.0 - one_hot) * cos_theta
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf detected in SphereFace output")
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
    def __init__(self, num_classes, feature_dim=1536, stage='stage1'):
        super().__init__()
        # self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone = models.efficientnet_b3(weights=None)

        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.stage = stage
        if stage == 'stage1':
            self.head = SphereFace(feature_dim, num_classes, m=Config.MARGIN)
        else:
            self.head = ClassificationHead(feature_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.dropout(x)
        if self.stage == 'stage1':
            return self.head(x, labels)
        return self.head(x)

class Trainer:
    def __init__(self, model, optimizer, class_weights=None):
        self.model = model.to(Config.DEVICE)
        self.optimizer = optimizer
        self.all_train_losses = []
        self.all_train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.num_train_batches_per_epoch = None
        self.global_iter = 0
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE) if class_weights is not None else None)

    def unfreeze_backbone(self, epoch):
        if epoch >= 2:
            for param in self.model.backbone.parameters():
                param.requires_grad = True

    def run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds = []
        all_labels = []
        
        pbar_desc = "Training" if is_train else "Validation"
        pbar = tqdm(loader, desc=pbar_desc)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"Batch {batch_idx} - Invalid inputs detected")
                continue

            self.optimizer.zero_grad()
            
            logits = self.model(inputs, labels if is_train and self.model.stage == 'stage1' else None)
            loss = self.criterion(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Batch {batch_idx} - Invalid loss")
                continue
                        
            if is_train:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
                if total_norm > 1.0:
                    print(f"Gradient norm clipped: {total_norm:.2f}")
                self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            iter_loss = loss.item()
            iter_accuracy = (preds == labels).float().mean().item()
            
            if is_train:
                self.all_train_losses.append(iter_loss)
                self.all_train_accs.append(iter_accuracy)
                self.global_iter += 1
            
            pbar.set_postfix({
                "Iter": self.global_iter if is_train else None,
                "Loss": iter_loss,
                "Acc": iter_accuracy
            })
        
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        if not is_train and len(all_labels) > 0:
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, epoch_acc, precision, recall, f1

def show_batch(loader):
    images, labels = next(iter(loader))
    mean = torch.tensor(processor.mean).view(3, 1, 1)
    std = torch.tensor(processor.std).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    plt.figure(figsize=(12, 8))
    for i in range(min(6, len(images))):
        plt.subplot(2, 3, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.savefig(f"{METRICS_DIR}/batch_visualization.png")
    plt.close()

def plot_metrics(trainer, metrics_dir):
    total_iters = len(trainer.all_train_losses)
    epochs = len(trainer.val_losses)
    num_train_batches_per_epoch = trainer.num_train_batches_per_epoch
    
    val_x = [(i+1) * num_train_batches_per_epoch for i in range(epochs)]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(total_iters), trainer.all_train_losses, label='Train', alpha=0.5)
    plt.plot(val_x, trainer.val_losses, label='Validation', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(total_iters), trainer.all_train_accs, label='Train', alpha=0.5)
    plt.plot(val_x, trainer.val_accs, label='Validation', marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_x, trainer.val_f1s, label='Validation', marker='o')
    plt.title('F1-Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/training_metrics.png")
    plt.close()

def setup_metrics_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f"{METRICS_DIR}/{timestamp}"
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir

def save_metrics_to_csv(trainer, metrics_dir):
    epochs = list(range(1, len(trainer.val_losses) + 1))
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': [sum(trainer.all_train_losses[i*trainer.num_train_batches_per_epoch:(i+1)*trainer.num_train_batches_per_epoch]) / trainer.num_train_batches_per_epoch for i in range(len(epochs))],
        'val_loss': trainer.val_losses,
        'val_acc': trainer.val_accs,
        'val_precision': trainer.val_precisions,
        'val_recall': trainer.val_recalls,
        'val_f1': trainer.val_f1s
    })
    metrics_df.to_csv(f"{metrics_dir}/epoch_metrics.csv", index=False)

def plot_confusion_matrix(model, test_loader, metrics_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    class_names = ['Alternaria leaf blight', 'Anthocyanosis', 'Anthracnose', 'Aphid', 'Aphid effects', 'Ascochyta blight', 'Bacterial spot', 'Black chaff', 'Black rot', 'Black spots', 'Blossom end rot', 'Botrytis cinerea', 'Burn', 'Downy mildew', 'Dry rot', 'Edema', 'Grey mold', 'Healthy', 'Late blight', 'Leaf deformation', 'Leaf miners', 'Loss of foliage turgor', 'Marginal leaf necrosis', 'Mealybug', 'Mechanical damage', 'Mosaic virus', 'Nutrient deficiency', 'Powdery mildew', 'Rust', 'Scale', 'Shot hole', 'Sooty mold', 'Spider mite', 'Thrips', 'Whitefly', 'Wilting', 'Yellow leaves']
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names[:Config.NUM_CLASSES], yticklabels=class_names[:Config.NUM_CLASSES])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    torch.cuda.empty_cache()
    
    metrics_dir = setup_metrics_dir()
    processor = DataProcessor()
    train_loader, val_loader, test_loader = processor.get_loaders()

    class_counts = torch.zeros(Config.NUM_CLASSES)
    for _, label in datasets.ImageFolder(DATA_DIR, transform=processor.val_transform):
        class_counts[label] += 1
    print(f"Class distribution: {class_counts}")
    
    class_weights = 1.0 / (class_counts.float() + 1e-6)  # Добавлен эпсилон для избежания деления на 0
    class_weights = class_weights / class_weights.sum() * Config.NUM_CLASSES
    
    show_batch(train_loader)

    # Stage 1: Обучение эмбеддингов с SphereFace
    model = DiseaseClassifier(num_classes=Config.NUM_CLASSES, feature_dim=Config.FEATURE_DIM, stage='stage1')
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3)
    trainer = Trainer(model, optimizer, class_weights)
    trainer.num_train_batches_per_epoch = len(train_loader)
    best_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS_STAGE1):
        trainer.unfreeze_backbone(epoch)
        train_loss, train_acc, _, _, _ = trainer.run_epoch(train_loader)
        val_loss, val_acc, val_precision, val_recall, val_f1 = trainer.run_epoch(val_loader, is_train=False)
        
        trainer.val_losses.append(val_loss)
        trainer.val_accs.append(val_acc)
        trainer.val_precisions.append(val_precision)
        trainer.val_recalls.append(val_recall)
        trainer.val_f1s.append(val_f1)
        
        print(f"Stage 1 Epoch {epoch+1}/{Config.NUM_EPOCHS_STAGE1}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | Prec: {val_precision:.2%} | Rec: {val_recall:.2%} | F1: {val_f1:.2%}")
        
        if not math.isnan(val_acc):
            scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}_stage1_best.pth")
            print(f"New best model saved to {MODEL_SAVE_PATH}_stage1_best.pth")
        
        save_metrics_to_csv(trainer, metrics_dir)
        plot_metrics(trainer, metrics_dir)

    # Stage 2: Обучение классификационной головы
    model = DiseaseClassifier(num_classes=Config.NUM_CLASSES, feature_dim=Config.FEATURE_DIM, stage='stage2')
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}_stage1_best.pth"), strict=False)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.features[-2:].parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.head.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    trainer = Trainer(model, optimizer, class_weights)
    trainer.num_train_batches_per_epoch = len(train_loader)
    trainer.all_train_losses, trainer.all_train_accs = [], []
    trainer.val_losses, trainer.val_accs = [], []
    trainer.val_precisions, trainer.val_recalls, trainer.val_f1s = [], [], []
    best_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS_STAGE2):
        train_loss, train_acc, _, _, _ = trainer.run_epoch(train_loader)
        val_loss, val_acc, val_precision, val_recall, val_f1 = trainer.run_epoch(val_loader, is_train=False)
        
        trainer.val_losses.append(val_loss)
        trainer.val_accs.append(val_acc)
        trainer.val_precisions.append(val_precision)
        trainer.val_recalls.append(val_recall)
        trainer.val_f1s.append(val_f1)
        
        print(f"Stage 2 Epoch {epoch+1}/{Config.NUM_EPOCHS_STAGE2}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | Prec: {val_precision:.2%} | Rec: {val_recall:.2%} | F1: {val_f1:.2%}")
        
        if not math.isnan(val_acc):
            scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}_stage2_best.pth")
            print(f"New best model saved to {MODEL_SAVE_PATH}_stage2_best.pth")
        
        save_metrics_to_csv(trainer, metrics_dir)
        plot_metrics(trainer, metrics_dir)

    test_loss, test_acc, test_precision, test_recall, test_f1 = trainer.run_epoch(test_loader, is_train=False)
    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2%} | Prec: {test_precision:.2%} | Rec: {test_recall:.2%} | F1: {test_f1:.2%}")
    plot_confusion_matrix(model, test_loader, metrics_dir)
    metrics_df = pd.read_csv(f"{metrics_dir}/epoch_metrics.csv")
    metrics_df.loc[len(metrics_df) - 1, 'test_loss'] = test_loss
    metrics_df.loc[len(metrics_df) - 1, 'test_acc'] = test_acc
    metrics_df.loc[len(metrics_df) - 1, 'test_precision'] = test_precision
    metrics_df.loc[len(metrics_df) - 1, 'test_recall'] = test_recall
    metrics_df.loc[len(metrics_df) - 1, 'test_f1'] = test_f1
    metrics_df.to_csv(f"{metrics_dir}/epoch_metrics.csv", index=False)