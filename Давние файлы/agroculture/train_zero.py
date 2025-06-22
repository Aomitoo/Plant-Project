import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from model_utils import Config, DataProcessor, DiseaseClassifier


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
    train_loader, test_loader = processor.get_loaders()
    
    model = DiseaseClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    trainer = Trainer(model, criterion, optimizer)

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc = trainer.run_epoch(train_loader)
        val_loss, val_acc = trainer.run_epoch(test_loader, is_train=False)
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")

    torch.save(model.state_dict(), "models/plant_disease_model.pth")
    print("Model saved successfully!")