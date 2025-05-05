import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow.keras import layers
import pathlib

# Download dataset
data_dir = pathlib.Path("D:\DISUES PLANT\DoctorP_dataset")

### CONFIG ###
BATCH_SIZE = 32
img_height = 128
img_width = 128

# Загрузка данных
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE)

# Получение имен классов
class_names = train_ds.class_names
num_classes = len(class_names)

# Базовые преобразования (изменение размера и масштабирование)
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(img_width, img_height),
    layers.Rescaling(1./255)
])

# Аугментация данных
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(1),  # 3% от максимального угла
    layers.RandomZoom(0.03),      # 3% зумирования
    layers.RandomContrast(0.001),  # Минимальное изменение контраста
    layers.RandomBrightness(0.001), # Минимальное изменение яркости
])

# Применение аугментации к тренировочным данным
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Предобработка для всех данных
train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x), y),
                       num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y),
                   num_parallel_calls=tf.data.AUTOTUNE)
augmented_train_ds = augmented_train_ds.map(lambda x, y: (resize_and_rescale(x), y),
                                          num_parallel_calls=tf.data.AUTOTUNE)

# Кеширование и предварительная загрузка для производительности
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
augmented_train_ds = augmented_train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Объединение оригинальных и аугментированных данных
final_train_ds = tf.data.Dataset.concatenate(train_ds, augmented_train_ds)

# # Визуализация примеров аугментации
# def visualize_augmentation(dataset, class_names):
#     plt.figure(figsize=(10, 10))
#     for images, labels in dataset.take(1):
#         for i in range(9):
#             ax = plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i].numpy())
#             plt.title(class_names[labels[i]])
#             plt.axis("off")
#     plt.show()

# # Визуализация аугментированных данных
# visualize_augmentation(augmented_train_ds, class_names)

# # Функция для обработки одиночного изображения
# def process_single_image(image_path):
#     # Загрузка и предобработка
#     image = Image.open(image_path).resize((img_width, img_height))
#     image_array = np.array(image) / 255.0  # Нормализация [0, 1]
    
#     # Добавление batch-размерности
#     image_batch = np.expand_dims(image_array, axis=0)
    
#     # Применение аугментации + гарантия диапазона
#     augmented = data_augmentation(image_batch)
    
#     return image_array, augmented[0]

# # Пример использования
# original, augmented = process_single_image('1526.jpg')

# # Визуализация
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(original)
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.title("Augmented")
# plt.imshow(augmented)
# plt.axis("off")
# plt.show()