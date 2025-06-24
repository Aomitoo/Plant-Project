import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import datetime
import numpy as np
import math

# Configuration class
class Config:
    BATCH_SIZE = 16
    NUM_CLASSES = 35
    IMG_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKBONE_FEATURE_DIM = 1536
    EMBEDDING_DIM = 1280
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3  # Adjusted for fine-tuning
    DATA_DIR = "plant_diseases/"  # Main dataset directory with class subdirs
    TRAIN_RATIO = 0.8  # 80% for train, 20% for val
    MODEL_PATH = "models/Efficient_70%_BEST copy.pth"

# Model definitions
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
        if labels is not None:
            print(f"Cos theta min: {cos_theta.min():.4f}, max: {cos_theta.max():.4f}")  # Отладка логиков
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

# Data transformations
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4569, 0.5046, 0.3590], std=[0.2169, 0.2138, 0.2120])
])

def setup_metrics_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f"metrics/{timestamp}_val"
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir

def plot_confusion_matrix(all_labels, all_preds, class_names, metrics_dir):
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 10})
    plt.title('Confusion Matrix (Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/val_confusion_matrix.png")
    plt.close()

def remove_misclassified_files(metrics_dir):
    txt_path = f"{metrics_dir}/misclassified_images.txt"
    if not os.path.exists(txt_path):
        print(f"No misclassified_images.txt found at {txt_path}. Skipping deletion.")
        return

    with open(txt_path, "r") as f:
        next(f)  # Skip header line
        for line in f:
            path = line.split(",")[0].strip()  # Extract file path
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
            else:
                print(f"File not found, skipping: {path}")

def train_and_validate():
    # Load the pre-trained model
    model = DiseaseClassifier(num_classes=Config.NUM_CLASSES, stage='stage1')
    state_dict = torch.load(Config.MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Load full dataset and split into train and val
    full_dataset = datasets.ImageFolder(Config.DATA_DIR, transform=transform)
    Config.NUM_CLASSES = len(full_dataset.classes)
    print(f"Detected classes: {full_dataset.classes}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    generator = torch.Generator().manual_seed(42)
    total_size = len(full_dataset)
    train_size = int(Config.TRAIN_RATIO * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    class_names = full_dataset.classes

    # Train for one epoch
    model.train()
    total_train_loss, total_train_correct, total_train_samples = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training (1 Epoch)")):
        try:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_train_correct += (preds == labels).sum().item()
            total_train_samples += inputs.size(0)
            iter_loss = loss.item()
            iter_accuracy = (preds == labels).float().mean().item() * 100  # Процент
            tqdm.write(f"Batch {batch_idx+1} | Train Loss: {iter_loss:.4f} | Acc: {iter_accuracy:.2f}%")
        except Exception as e:
            print(f"Error during training: {e}. Skipping batch.")

    train_loss = total_train_loss / total_train_samples
    train_acc = total_train_correct / total_train_samples * 100
    print(f"Training finished | Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    # Evaluate on validation set
    model.eval()
    total_val_loss, total_val_correct, total_val_samples = 0.0, 0, 0
    all_preds = []
    all_labels = []
    misclassified = []
    image_paths = [sample[0] for sample in full_dataset.samples]

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            try:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                total_val_correct += (preds == labels).sum().item()
                total_val_samples += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                iter_loss = loss.item()
                iter_accuracy = (preds == labels).float().mean().item() * 100  # Процент
                tqdm.write(f"Batch {batch_idx+1} | Val Loss: {iter_loss:.4f} | Acc: {iter_accuracy:.2f}%")

                # Track misclassified images
                for i, (pred, true) in enumerate(zip(preds, labels)):
                    if pred != true:
                        global_idx = batch_idx * Config.BATCH_SIZE + i
                        if global_idx < len(image_paths):
                            misclassified.append((image_paths[global_idx], true.item(), pred.item()))
            except Exception as e:
                print(f"Error during validation: {e}. Skipping batch.")

    val_loss = total_val_loss / total_val_samples
    val_acc = total_val_correct / total_val_samples * 100
    print(f"Validation finished | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

    # Generate confusion matrix
    metrics_dir = setup_metrics_dir()
    plot_confusion_matrix(all_labels, all_preds, class_names, metrics_dir)

    # Save misclassified images to text file
    with open(f"{metrics_dir}/misclassified_images.txt", "w") as f:
        f.write("Path,True Label,Predicted Label\n")
        for path, true, pred in misclassified:
            f.write(f"{path},{class_names[true]},{class_names[pred]}\n")

    # Remove misclassified files
    # remove_misclassified_files(metrics_dir)

    print(f"Confusion matrix saved to {metrics_dir}/val_confusion_matrix.png")
    print(f"Misclassified images logged to {metrics_dir}/misclassified_images.txt")

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    train_and_validate()