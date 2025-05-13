import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models_utils_new import Config, DataProcessor, DiseaseClassifier


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(Config.DEVICE)
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

    def run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, correct = 0.0, 0
        pbar = tqdm(loader, desc="Training" if is_train else "Validation")
        
        for inputs, labels in pbar:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            logits = self.model(inputs, labels)
            loss = F.cross_entropy(logits, labels)

            if is_train:
                loss.backward()
                self.optimizer.step()
            # with torch.cuda.amp.autocast():
            #     logits = self.model(inputs, labels)
            #     loss = F.cross_entropy(logits, labels)
            
            # if is_train:
            #     self.scaler.scale(loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            pbar.set_postfix({"Loss": loss.item()})

        return total_loss / len(loader), correct / len(loader.dataset)


if __name__ == "__main__":
    processor = DataProcessor()
    train_loader, test_loader = processor.get_loaders()
    
    # Проверка размеров датасетов
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Total images: {len(train_loader.dataset) + len(test_loader.dataset)}")
    print("Пример меток:", next(iter(train_loader))[1])

    model = DiseaseClassifier()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    trainer = Trainer(model, optimizer)

    best_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = trainer.run_epoch(train_loader)
        val_loss, val_acc = trainer.run_epoch(test_loader, is_train=False)
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/doctorp_resnext_cosface.pth")
            print(f"New best model saved! Acc: {best_acc:.2%}")

    print("Training completed!")