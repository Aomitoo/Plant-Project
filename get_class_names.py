from models_utils_new import Config, DataProcessor
import os

processor = DataProcessor()
full_dataset = processor.get_loaders()[0].dataset.dataset  # Получаем полный датасет
class_names = full_dataset.classes  # Список названий классов
print(class_names)  # Выводит список в порядке индексов