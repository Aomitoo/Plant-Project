import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from model_utils import Config, DataProcessor, DiseaseClassifier
import os


def save_checkpoint(epoch, model, optimizer, val_loss):
    """Сохраняет чекпоинт модели и оптимизатора"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    # Создаем папку, если ее нет
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Путь для сохранения
    checkpoint_path = Config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer):
    """Загружает последний чекпоинт, если он существует"""
    checkpoints = list(Config.CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return 0, float('inf')
    
    # Загружаем метаданные классов
    processor = DataProcessor()
    saved_classes = processor.load_classes()
    num_classes = len(saved_classes)
    
    # Меняем последний слой модели при необходимости
    if model.base_model.fc.out_features != num_classes:
        print(f"Обновление слоя: {model.base_model.fc.out_features} -> {num_classes}")
        num_ftrs = model.base_model.fc.in_features
        model.base_model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Загружаем чекпоинт
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint, map_location=Config.DEVICE, weights_only=True)
    
    # Частичная загрузка весов
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'] + 1, checkpoint['val_loss']


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model.to(Config.DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

    def run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, correct = 0.0, 0
        pbar = tqdm(loader, desc="Training" if is_train else "Validation")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            pbar.set_postfix({"Loss": loss.item()})

        return total_loss / len(loader), correct / len(loader.dataset)


if __name__ == "__main__":
    processor = DataProcessor()
    
    # Загрузка данных с автоматическим разделением
    train_loader, val_loader, test_loader = processor.get_loaders(use_new_data=True)

    # Проверка разделения
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset) if test_loader else 0} samples")

    # Получаем актуальные классы
    class_names = processor.get_all_classes()  # Метод из предыдущего ответа
    processor.save_classes(class_names)
    
    # Инициализация модели с правильным числом классов
    model = DiseaseClassifier(num_classes=len(class_names))
    model.to(Config.DEVICE)

    # Загрузка предобученных весов
    try:
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
        print("Loaded pretrained model!")
    except:
        print("Training from scratch!")
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # Загрузка чекпоинта
    start_epoch, best_val_loss = load_checkpoint(model, optimizer)
    
    # Обучение
    trainer = Trainer(model, nn.CrossEntropyLoss(), optimizer)

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = trainer.run_epoch(train_loader)
        val_loss, val_acc = trainer.run_epoch(test_loader, is_train=False)

        # Сохраняем чекпоинт
        save_checkpoint(epoch, model, optimizer, val_loss)

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with val loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}\n")

