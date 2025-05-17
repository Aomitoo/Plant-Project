import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models_utils_new import Config, DataProcessor, DiseaseClassifier, ArcFace
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os
import datetime
import seaborn as sns


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(Config.DEVICE)
        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler("cuda")
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.precision = []
        self.recall = []
        self.f1 = []

        
        # Замораживаем backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        # Размораживаем слои для обучения
        for param in self.model.embedding.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True


    def run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, correct = 0.0, 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc="Training" if is_train else "Validation")
        
        for inputs, labels in pbar: 
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            pe
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(inputs, labels)
                loss = F.cross_entropy(logits, labels)
            
            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"Loss": loss.item()})

        # Счет метрик
        accuracy = correct / len(loader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return total_loss / len(loader), accuracy, precision, recall, f1
    
def show_batch(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
    plt.show()

def plot_metrics(trainer, metrics_dir):
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(trainer.train_losses, label='Train')
    plt.plot(trainer.val_losses, label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График точности
    plt.subplot(1, 3, 2)
    plt.plot(trainer.train_accs, label='Train')
    plt.plot(trainer.val_accs, label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График F1-Score
    plt.subplot(1, 3, 3)
    plt.plot(trainer.f1, label='Weighted F1')
    plt.title('F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/training_metrics.png")
    plt.close()

def setup_metrics_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = f"metrics/{timestamp}"
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir

def save_metrics_to_csv(trainer, metrics_dir):
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(trainer.train_losses)+1)),
        'train_loss': trainer.train_losses,
        'val_loss': trainer.val_losses,
        'train_acc': trainer.train_accs,
        'val_acc': trainer.val_accs,
        'precision': trainer.precision,
        'recall': trainer.recall,
        'f1': trainer.f1
    })
    metrics_df.to_csv(f"{metrics_dir}/metrics.csv", index=False)

# Матрица ошибок
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
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{metrics_dir}/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    metrics_dir = setup_metrics_dir()
    processor = DataProcessor()
    train_loader, test_loader = processor.get_loaders()
    
    # Проверка размеров датасетов
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Total images: {len(train_loader.dataset) + len(test_loader.dataset)}")
    print("Пример меток:", next(iter(train_loader))[1])
     
    show_batch(train_loader)

    model = DiseaseClassifier()
    optimizer = optim.AdamW(
    [
        {'params': model.embedding.parameters()},
        {'params': model.head.parameters()}
    ],
    lr=Config.LR,  
    weight_decay=1e-4
)
    
    for layer in list(model.backbone.children())[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=2, 
        factor=0.1
    )
    trainer = Trainer(model, optimizer)

    best_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc, train_prec, train_rec, train_f1 = trainer.run_epoch(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = trainer.run_epoch(test_loader, is_train=False)
        
        # Сохраняем метрики
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.train_accs.append(train_acc)
        trainer.val_accs.append(val_acc)
        trainer.precision.append(val_prec)
        trainer.recall.append(val_rec)
        trainer.f1.append(val_f1)
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | Prec: {train_prec:.2f} | Rec: {train_rec:.2f} | F1: {train_f1:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | Prec: {val_prec:.2f} | Rec: {val_rec:.2f} | F1: {val_f1:.2f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/doctorp_resnext_arcface.pth")
            print(f"New best model saved!")
    
    plot_metrics(trainer, metrics_dir)
    plot_confusion_matrix(model, test_loader, metrics_dir)
    save_metrics_to_csv(trainer, metrics_dir)