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
from rembg import remove
import cv2

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

def is_valid_image(image):
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    color_pixels = np.any(img_array[:, :, :3] > 10, axis=2)
    color_ratio = np.sum(color_pixels) / (height * width)
    rgb_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hue_variance = np.var(hue_hist)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = height * width
    area_ratio = contour_area / total_area
    is_valid = (color_ratio > 0.5 and area_ratio > 0.1 and hue_variance > 15)
    if area_ratio > 0.8 and hue_variance > 5:
        return False
    return is_valid

class RemoveBackground:
    def __call__(self, img):
        img_no_bg = remove(img)
        if is_valid_image(img_no_bg):
            background = Image.new("RGB", img_no_bg.size, (255, 255, 255))
            background.paste(img_no_bg, mask=img_no_bg.split()[3])
            return background
        else:
            return img.convert("RGB")

transform = transforms.Compose([
    # RemoveBackground(),
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4569, 0.5046, 0.3590], std=[0.2169, 0.2138, 0.2120])
])

def setup_metrics_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f"metrics/{timestamp}_test"
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir

def evaluate_test_set(test_dir="test_data/"):
    metrics_dir = setup_metrics_dir()
    model = DiseaseClassifier(num_classes=Config.NUM_CLASSES, stage='stage1')
    state_dict = torch.load('models/BEST_efficientnet_sphereface_ 85.25% copy.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(Config.DEVICE)

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    class_names = test_dataset.classes

    misclassified = []
    image_paths = [sample[0] for sample in test_dataset.samples]

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating Test Set")):
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

                for i, (pred, true) in enumerate(zip(preds, labels)):
                    if pred != true:
                        global_idx = batch_idx * Config.BATCH_SIZE + i
                        if global_idx < len(image_paths):
                            misclassified.append((image_paths[global_idx], true.item(), pred.item()))

                if batch_idx in [0, 4, 7]:
                    torch.save({'inputs': inputs.cpu(), 'labels': labels.cpu()}, f"{metrics_dir}/batch_{batch_idx}.pt")
                    torchvision.utils.save_image(inputs.cpu(), f"{metrics_dir}/batch_{batch_idx}.png", nrow=int(inputs.size(0) ** 0.5), normalize=True)
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

    with open(f"{metrics_dir}/misclassified_images.txt", "w") as f:
        f.write("Path,True Label,Predicted Label\n")
        for path, true, pred in misclassified:
            f.write(f"{path},{class_names[true]},{class_names[pred]}\n")
    print(f"Misclassified images logged to {metrics_dir}/misclassified_images.txt")

    plot_metrics(test_acc, test_loss, test_precision, test_recall, test_f1, metrics_dir)
    plot_confusion_matrix(all_labels, all_preds, class_names, metrics_dir)

    return test_acc

def plot_metrics(test_acc, test_loss, test_precision, test_recall, test_f1, metrics_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Test Accuracy'], [test_acc], color='green')
    plt.title('Test Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
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

if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    evaluate_test_set()