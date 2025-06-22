from rembg import remove
from PIL import Image
import os
import numpy as np
import cv2

DATA_DIR = "plant_diseases/"
OUTPUT_DIR = "plant_diseases_no_bg/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid_image(image):
    # Преобразуем в массив
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Проверяем цветные пиксели (не черные и не полностью прозрачные)
    color_pixels = np.any(img_array[:, :, :3] > 10, axis=2)  # Игнорируем альфа-канал
    color_ratio = np.sum(color_pixels) / (height * width)
    
    # Анализ гистограммы для проверки разнообразия цветов (пятна)
    rgb_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)  # Сначала RGBA -> RGB
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hue_variance = np.var(hue_hist)
    
    # Проверяем контуры листа
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False  # Полностью черное или пустое изображение
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = height * width
    area_ratio = contour_area / total_area
    
    # Условия для "хорошего" изображения
    # Минимум 5% цветных пикселей, контур > 10% площади, и вариация оттенков (пятна)
    is_valid = (color_ratio > 0.5 and area_ratio > 0.1 and hue_variance > 15)
    
    # Если лист занимает почти всё изображение (>90%), но есть вариации (пятна), сохраняем
    if area_ratio > 0.8 and hue_variance > 5:
        return False
    
    return is_valid

for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            input_path = os.path.join(class_dir, img_name)
            output_path = os.path.join(output_class_dir, os.path.splitext(img_name)[0] + '.jpg')  # Изменено на .jpg
            input_image = Image.open(input_path)
            output_image = remove(input_image)
            # Проверяем качество изображения
            if is_valid_image(output_image):
                # Конвертируем RGBA в RGB с белым фоном
                background = Image.new("RGB", output_image.size, (255, 255, 255))
                background.paste(output_image, mask=output_image.split()[3])  # Альфа-канал как маска
                background.save(output_path, quality=95)  # Сохраняем как JPG с высоким качеством
            else:
                print(f"Skipped {img_name}: Poor quality or spots removed detected.")
                # Сохраняем оригинал как JPG
                rgb_input = input_image.convert("RGB")
                rgb_input.save(output_path, quality=95)