import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import warnings
from torch.amp import autocast, GradScaler  
from tqdm import tqdm
warnings.filterwarnings('ignore') 


# Конфигурация 
class Config:
    """Central configuration class"""
    DATA_DIR = Path("data")                    # Путь к данным
    BATCH_SIZE = 64                            # Размер батча
    NUM_EPOCHS = 15                            # Количество эпох
    LR = 0.001                                 # Скорость обучения
    NUM_CLASSES = 38                           # Количество классов
    IMG_SIZE = 256                             # Размер изображения
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)


# Обработка данных 
class DataProcessor:
    """Обработка и аугментация данных"""
    def __init__(self):
        # Базовые преобразования для тестовых данных
        self.test_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

        # Аугментация для тренировочных данных
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def get_loaders(self):
        """Создает DataLoader для тренировочных и тестовых данных"""
        train_data = datasets.ImageFolder(
            Config.DATA_DIR / "train",
            transform=self.train_transform
        )
        test_data = datasets.ImageFolder(
            Config.DATA_DIR / "valid",
            transform=self.test_transform
        )

        return (
    DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    DataLoader(test_data, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
)


# Модель 
class DiseaseClassifier(nn.Module):
    """Классификатор болезней растений"""
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet34(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        return self.base_model(x)


# Тренировочный цикл 
class Trainer:
    """Класс для обучения модели"""
    def __init__(self, model, criterion, optimizer):
        self.model = model.to(Config.DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def run_epoch(self, loader, is_train=True):
        """Запуск одной эпохи обучения/валидации"""
        self.model.train(is_train)
        total_loss = 0.0
        correct = 0
        
        pbar = tqdm(loader, desc="Обучение" if is_train else "Валидация")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)


            if is_train:
                self.optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Для валидации обычный backward (без mixed precision)
                loss.backward()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            
            pbar.set_postfix({"Loss": loss.item()})

        return total_loss / len(loader), correct / len(loader.dataset)


#  Утилиты
def predict(model, image_path, class_names):
    """Предсказание для одного изображения """
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(1).item()
    
    return class_names[pred_idx]

print("S")
# Основной пайплайн 
if __name__ == "__main__":
    # Инициализация компонентов
    processor = DataProcessor()
    print("1 step")
    train_loader, test_loader = processor.get_loaders()
    print("2 step")
    
    model = DiseaseClassifier()
    print("3 step")
    criterion = nn.CrossEntropyLoss()
    print("4 step")
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    print("5 step")
    trainer = Trainer(model, criterion, optimizer)

    print("Количество тренировочных изображений:", len(train_loader.dataset))
    print("Количество тестовых изображений:", len(test_loader.dataset))
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    print(f"Название GPU: {torch.cuda.get_device_name(0)}")

    print("Обучение старт step")
    # Обучение
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nЭпоха {epoch+1}/{Config.NUM_EPOCHS}")
        train_loss, train_acc = trainer.run_epoch(train_loader)
        print(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
    
        val_loss, val_acc = trainer.run_epoch(test_loader, is_train=False)
        print(f"[Val] Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
    
    print("End")
    # Сохранение весов ()
    torch.save(model.state_dict(), "models/plant_disease_model.pth")

    # Пример использования
    model = DiseaseClassifier()
    model.load_state_dict(torch.load("models/plant_disease_model.pth"))
    model.to(Config.DEVICE)

    class_names = train_loader.dataset.classes
    print("Пример предсказания:", predict(model, "data/test/AppleCedarRust1.jpg", class_names))