import torch
import os
from pathlib import Path
from tqdm import tqdm
from model_utils import Config, DiseaseClassifier, predict, DataProcessor
from torchvision.datasets import ImageFolder

# Disease-to-Pesticide mapping (simplified example, can be expanded or customized)
disease_to_pesticide = {
    "Apple___Apple_scab": "Captan, Mancozeb",
    "Apple___Black_rot": "Ziram, Captan",
    "Apple___Cedar_apple_rust": "Myclobutanil",
    "Apple___healthy": "No pesticide needed",
    "Blueberry___healthy": "No pesticide needed",
    "Cherry_(including_sour)___Powdery_mildew": "Sulfur, Myclobutanil",
    "Cherry_(including_sour)___healthy": "No pesticide needed",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Strobilurins",
    "Corn_(maize)___Common_rust_": "Propiconazole",
    "Corn_(maize)___Northern_Leaf_Blight": "Azoxystrobin",
    "Corn_(maize)___healthy": "No pesticide needed",
    "Grape___Black_rot": "Mancozeb, Captan",
    "Grape___Esca_(Black_Measles)": "No effective chemical control",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Mancozeb",
    "Grape___healthy": "No pesticide needed",
    "Orange___Haunglongbing_(Citrus_greening)": "Vector control: Imidacloprid",
    "Peach___Bacterial_spot": "Copper-based sprays",
    "Peach___healthy": "No pesticide needed",
    "Pepper,_bell___Bacterial_spot": "Copper-based sprays",
    "Pepper,_bell___healthy": "No pesticide needed",
    "Potato___Early_blight": "Chlorothalonil, Mancozeb",
    "Potato___Late_blight": "Metalaxyl, Mancozeb",
    "Potato___healthy": "No pesticide needed",
    "Raspberry___healthy": "No pesticide needed",
    "Soybean___healthy": "No pesticide needed",
    "Squash___Powdery_mildew": "Sulfur, Neem oil",
    "Strawberry___Leaf_scorch": "Captan",
    "Strawberry___healthy": "No pesticide needed",
    "Tomato___Bacterial_spot": "Copper sprays",
    "Tomato___Early_blight": "Chlorothalonil, Mancozeb",
    "Tomato___Late_blight": "Metalaxyl",
    "Tomato___Leaf_Mold": "Copper-based fungicides",
    "Tomato___Septoria_leaf_spot": "Chlorothalonil",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Insecticidal soap, Neem oil",
    "Tomato___Target_Spot": "Chlorothalonil",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Vector control: Imidacloprid",
    "Tomato___Tomato_mosaic_virus": "Sanitation, resistant varieties",
    "Tomato___healthy": "No pesticide needed"
}


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
    
    print(f"| {'Файл':<30} | {'Предсказанный класс':<28} | Пестициды")
    print("--")

    for filename, pred in results:
        # print(f"| {filename:<30} | {pred:<28} | {disease_to_pesticide[pred]} ")
        print(f"| {filename:<30} | {pred:<28} |")
 

if __name__ == "__main__":
    # Инициализация DataProcessor для получения всех классов
    processor = DataProcessor()
    
    # Получение актуальных классов из объединённого датасета
    class_names = processor.load_classes()
    if not class_names:
        raise ValueError("Не найдены классы в models/classes.txt")
    
    # Инициализация модели с правильным числом классов
    model = DiseaseClassifier(num_classes=len(class_names))
    
    # Безопасная загрузка весов
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=True),
        strict=False
    )
    model.to(Config.DEVICE)
    model.eval()

    # Путь к тестовым изображениям
    test_dir = Config.NEW_DATA_DIR if (Config.NEW_DATA_DIR / "test").exists() else Config.DATA_DIR / "test"
    
    # Анализ всех изображений
    analyze_test_images(model, test_dir, class_names)