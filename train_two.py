import torch
import torchvision.models as models
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import time


# Настройка логирования
def log_message(message):
    with open("training_log.txt", "a") as f:
        f.write(message + "\n")
    print(message)

# Обновленный способ загрузки модели
def get_model(device):
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    model = models.resnext50_32x4d(weights=weights)
    model.fc = torch.nn.Identity()
    return model.to(device)


model = models.resnext50_32x4d(pretrained=True)
# Удаляем классификационный слой
model.fc = torch.nn.Identity()  # Теперь модель возвращает эмбеддинги

class ArcFaceModel(torch.nn.Module):
    def __init__(self, backbone, num_classes, embedding_size=512, device='cuda'):
        super().__init__()
        self.backbone = backbone.to(device)
        self.fc = torch.nn.Linear(embedding_size, num_classes)  # Классификационный слой
        self.arcface = ArcFaceLoss(num_classes, embedding_size)  # ArcFace функция потерь

    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        logits = self.fc(embeddings)
        if labels is not None:
            loss = self.arcface(logits, labels)
            return loss
        return logits

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.5, scale=32):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weights = torch.nn.Parameter(torch.randn(embedding_size, num_classes))

    def forward(self, logits, labels):
        # Нормализация весов и эмбеддингов
        normalized_weights = torch.nn.functional.normalize(self.weights, dim=0)
        normalized_embeddings = torch.nn.functional.normalize(logits, dim=1)

        # Вычисление косинусов углов
        cos_theta = torch.mm(normalized_embeddings, normalized_weights)
        theta = torch.acos(torch.clamp(cos_theta, -1, 1))

        # Добавление углового запаса
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.weights.shape[1])
        cos_theta_m = torch.cos(theta + self.margin * one_hot)

        # Масштабирование
        logits = self.scale * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss


class DataProcessor:
    def __init__(self):
        # Параметры нормализации из исследования
        self.mean = [0.4467, 0.4889, 0.3267]
        self.std = [0.2299, 0.2224, 0.2289]
        
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomRotation(45),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.05, contrast=0.05),
            # transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.6, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
           
        ])

    def get_loaders(self):
        # Загрузка всего датасета
        full_dataset = datasets.ImageFolder(
            Path("D:/DISUES PLANT/DoctorP_dataset"),
            transform=self.test_transform  # Базовые преобразования для всего датасета
        )
        
        # Ручное перемешивание перед разделением
        generator = torch.Generator().manual_seed(42)  # Для воспроизводимости
        
        # Разделение 80/20 с сохранением баланса классов
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=generator
        )
        
        # Применяем аугментации только к тренировочному набору
        train_dataset.dataset.transform = self.train_transform

        train_loader = DataLoader(
            train_dataset, 
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader


if __name__ == '__main__':
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")

    # Инициализация компонентов
    model = ArcFaceModel(
        backbone=get_model(device), 
        num_classes=68,
        device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    processor = DataProcessor()
    train_loader, test_loader = processor.get_loaders()

    # Цикл обучения
    for epoch in range(10):
        start_time = time.time()
        train_loss = 0.0
        
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Проверка устройств
            assert next(model.parameters()).is_cuda == images.is_cuda
            
            loss = model(images, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
        
        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Логирование
        epoch_time = time.time() - start_time
        avg_loss = train_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        log_message(
            f"Epoch {epoch+1} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Test Acc: {accuracy:.2f}%"
        )

    log_message("Training completed!")