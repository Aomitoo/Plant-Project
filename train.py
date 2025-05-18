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
import numpy as np

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(Config.DEVICE)
        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler("cuda")
        self.all_train_losses = []
        self.all_train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.num_train_batches_per_epoch = None
        self.global_iter = 0

        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        for param in self.model.embedding.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
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
            
            if is_train:
                self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16 ):
                logits = self.model(inputs, labels)
                loss = F.cross_entropy(logits, labels)
            
            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Логирование нормы градиентов
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if total_norm > 1.0:
                    print(f"Gradient norm clipped: {total_norm}")
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
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
    mean = torch.tensor([0.4467, 0.4889, 0.3267]).view(3, 1, 1)
    std = torch.tensor([0.2299, 0.2224, 0.2289]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.savefig('batch_visualization.png')
    plt.close()

def plot_metrics(trainer, metrics_dir):
    total_iters = len(trainer.all_train_losses)
    epochs = len(trainer.val_losses)
    num_train_batches_per_epoch = trainer.num_train_batches_per_epoch
    
    # Вычисляем x-координаты для валидационных метрик (конец каждой эпохи)
    val_x = [ (i+1) * num_train_batches_per_epoch for i in range(epochs) ]
    
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(range(total_iters), trainer.all_train_losses, label='Train', alpha=0.5)
    plt.plot(val_x, trainer.val_losses, label='Validation', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    # График точности
    plt.subplot(1, 3, 2)
    plt.plot(range(total_iters), trainer.all_train_accs, label='Train', alpha=0.5)
    plt.plot(val_x, trainer.val_accs, label='Validation', marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График F1-меры (только для валидации)
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
    metrics_dir = f"metrics/{timestamp}"
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir

def save_metrics_to_csv(trainer, metrics_dir):
    epochs = list(range(1, len(trainer.val_losses) + 1))
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': [sum(trainer.all_train_losses[i*trainer.num_train_batches_per_epoch:(i+1)*trainer.num_train_batches_per_epoch]) / trainer.num_train_batches_per_epoch for i in range(len(epochs))],
        'val_loss': trainer.val_losses,
        'train_acc': [sum([acc * bs for acc, bs in zip(trainer.all_train_accs[i*trainer.num_train_batches_per_epoch:(i+1)*trainer.num_train_batches_per_epoch], [Config.BATCH_SIZE]*trainer.num_train_batches_per_epoch)]) / (Config.BATCH_SIZE * trainer.num_train_batches_per_epoch) for i in range(len(epochs))],
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
    
    # for layer in list(model.backbone.children())[-3:]:
    #     for param in layer.parameters():
    #         param.requires_grad = True


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=2, 
        factor=0.1
    )
    trainer = Trainer(model, optimizer)
    trainer.num_train_batches_per_epoch = len(train_loader)

    best_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        _ = trainer.run_epoch(train_loader)
        val_loss, val_acc, val_precision, val_recall, val_f1 = trainer.run_epoch(test_loader, is_train=False)
        
        trainer.val_losses.append(val_loss)
        trainer.val_accs.append(val_acc)
        trainer.val_precisions.append(val_precision)
        trainer.val_recalls.append(val_recall)
        trainer.val_f1s.append(val_f1)
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {sum(trainer.all_train_losses[-trainer.num_train_batches_per_epoch:]) / trainer.num_train_batches_per_epoch:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | Prec: {val_precision:.2%} | Rec: {val_recall:.2%} | F1: {val_f1:.2%}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/doctorp_resnext_arcface.pth")
            print(f"New best model saved!")
        
        save_metrics_to_csv(trainer, metrics_dir)
        plot_metrics(trainer, metrics_dir)
    
    plot_confusion_matrix(model, test_loader, metrics_dir)