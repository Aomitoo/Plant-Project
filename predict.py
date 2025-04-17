import torch
import os
from pathlib import Path
from tqdm import tqdm
from model_utils import Config, DiseaseClassifier, predict
from torchvision.datasets import ImageFolder


def analyze_test_images(model, test_dir, class_names):
    # Получаем все файлы изображений в тестовой директории
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']
    image_files = [
        f for f in os.listdir(test_dir)
        if os.path.splitext(f)[1] in image_extensions
    ]

    print(f"\n🔍 Найдено {len(image_files)} изображений для анализа:")
    
    results = []
    for filename in tqdm(image_files, desc="Обработка изображений"):
        image_path = os.path.join(test_dir, filename)
        try:
            prediction = predict(model, image_path, class_names)
            results.append((filename, prediction))
        except Exception as e:
            results.append((filename, f"Ошибка: {str(e)}"))

    # Вывод результатов в табличном формате
    print("\n📊 Результаты предсказаний:")
    print("-" * 65)
    print(f"| {'Файл':<30} | {'Предсказанный класс':<28} |")
    print("-" * 65)
    for filename, pred in results:
        print(f"| {filename:<30} | {pred:<28} |")
    print("-" * 65)


if __name__ == "__main__":
    # Загрузка модели
    model = DiseaseClassifier()
    model.load_state_dict(torch.load("models/plant_disease_model.pth"))
    model.to(Config.DEVICE)
    model.eval()

    # Получение названий классов
    train_dataset = ImageFolder(Config.DATA_DIR / "train")
    class_names = train_dataset.classes

    # Путь к тестовым изображениям
    test_dir = Config.DATA_DIR / "test"
    
    # Анализ всех изображений
    analyze_test_images(model, test_dir, class_names)