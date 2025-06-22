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
            if batch_idx == 59:
                print(f"Batch {batch_idx} - Inputs min: {torch.min(inputs)}, max: {torch.max(inputs)}")
       
            self.optimizer.zero_grad()
            
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = self.model(inputs, labels)
            if torch.isnan(logits).any():
                print(f"Batch {batch_idx} - Inputs shape: {inputs.shape}")
                print(f"Batch {batch_idx} - Inputs min: {torch.min(inputs)}, max: {torch.max(inputs)}")
                print(f"Batch {batch_idx} - Labels: {labels}")
                # Визуализация первого изображения в батче
                import matplotlib.pyplot as plt
                img = inputs[0].cpu().permute(1, 2, 0).numpy()  # Переводим в [H, W, C]
                plt.imshow(img)
                plt.title(f"Batch {batch_idx}, Image 0")
                plt.savefig(f"batch_59_image_0.png")
                plt.close()
                print(f"Nan in logits at epoch {epoch}, batch {batch_idx}")
            loss = F.cross_entropy(logits, labels)
            if torch.isnan(loss):
                print(f"Batch {batch_idx} - Inputs shape: {inputs.shape}")
                print(f"Batch {batch_idx} - Inputs min: {torch.min(inputs)}, max: {torch.max(inputs)}")
                print(f"Batch {batch_idx} - Labels: {labels}")
                # Визуализация первого изображения в батче
                import matplotlib.pyplot as plt
                img = inputs[0].cpu().permute(1, 2, 0).numpy()  # Переводим в [H, W, C]
                plt.imshow(img)
                plt.title(f"Batch {batch_idx}, Image 0")
                plt.savefig(f"batch_59_image_0.png")
                plt.close()
                print(f"Nan in loss at epoch {epoch}, batch {batch_idx}")
                        
            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Логирование нормы градиентов
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
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
    mean = torch.tensor([0.4432, 0.4937, 0.3295]).view(3, 1, 1)
    std = torch.tensor([0.1955, 0.1907, 0.1878]).view(3, 1, 1)
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
    metrics_dir = f"metrics 5-fold cross-validation/{timestamp}"
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
    
    # Define class names (replace with your actual class names)
    class_names = ['Alternaria leaf blight', 'Anthocyanosis', 'Anthracnose', 'Ants', 'Aphid', 'Aphid effects', 'Ascochyta blight', 'Bacterial spot', 'Black chaff', 'Black rot', 'Black spots', 'Blossom end rot', 'Botrytis cinerea', 'Burn', 'Canker', 'Caterpillars', 'Cherry leaf spot', 'Coccomyces of pome fruits', 'Colorado beetle', 'Colorado beetle effects', 'Corn downy mildew', 'Cyclamen mite', 'Downy mildew', 'Dry rot', 'Edema', 'Esca', 'Eyespot', 'Frost cracks', 'Galls', 'Grey mold', 'Gryllotalpa', 'Gryllotalpa effects', 'Healthy', 'Late blight', 'Leaf deformation', 'Leaf miners', 'Leaf spot']
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    metrics_dir = setup_metrics_dir()
    processor = DataProcessor()
    
    # Lists to store metrics across folds
    all_val_losses = []
    all_val_accs = []
    all_val_precisions = []
    all_val_recalls = []
    all_val_f1s = []

    for fold, (train_loader, val_loader) in enumerate(processor.get_loaders()):
        print(f"\nFold {fold + 1}/5")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print("Пример меток:", next(iter(train_loader))[1])
        
        show_batch(train_loader)

        model = DiseaseClassifier()
        optimizer = optim.AdamW(
            [
                {'params': model.embedding.parameters()},
                {'params': model.head.parameters()},
                {'params': model.classifier.parameters()}
            ],
            lr=Config.LR,  
            weight_decay=1e-2
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
        trainer.num_train_batches_per_epoch = len(train_loader)

        best_acc = 0.0
        fold_val_losses = []
        fold_val_accs = []
        fold_val_precisions = []
        fold_val_recalls = []
        fold_val_f1s = []

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_acc, _, _, _ = trainer.run_epoch(train_loader)
            val_loss, val_acc, val_precision, val_recall, val_f1 = trainer.run_epoch(val_loader, is_train=False)
            
            trainer.val_losses.append(val_loss)
            trainer.val_accs.append(val_acc)
            trainer.val_precisions.append(val_precision)
            trainer.val_recalls.append(val_recall)
            trainer.val_f1s.append(val_f1)
            
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            fold_val_precisions.append(val_precision)
            fold_val_recalls.append(val_recall)
            fold_val_f1s.append(val_f1)

            print(f"Fold {fold + 1} Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | Prec: {val_precision:.2%} | Rec: {val_recall:.2%} | F1: {val_f1:.2%}")
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"{metrics_dir}/doctorp_resnext_arcface_fold{fold + 1}.pth")
                print(f"New best model saved for fold {fold + 1}!")

            save_metrics_to_csv(trainer, metrics_dir)
            plot_metrics(trainer, metrics_dir)

        all_val_losses.append(fold_val_losses)
        all_val_accs.append(fold_val_accs)
        all_val_precisions.append(fold_val_precisions)
        all_val_recalls.append(fold_val_recalls)
        all_val_f1s.append(fold_val_f1s)

    # Compute and print average metrics across folds
    avg_val_losses = np.mean(all_val_losses, axis=0)
    avg_val_accs = np.mean(all_val_accs, axis=0)
    avg_val_precisions = np.mean(all_val_precisions, axis=0)
    avg_val_recalls = np.mean(all_val_recalls, axis=0)
    avg_val_f1s = np.mean(all_val_f1s, axis=0)

    print("\nAverage Metrics Across Folds:")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}: Val Loss: {avg_val_losses[epoch]:.4f} | Acc: {avg_val_accs[epoch]:.2%} | Prec: {avg_val_precisions[epoch]:.2%} | Rec: {avg_val_recalls[epoch]:.2%} | F1: {avg_val_f1s[epoch]:.2%}")