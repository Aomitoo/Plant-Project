import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import PIL
import PIL.Image

from tensorflow.keras import layers

import pathlib

# Download dataset
data_dir = pathlib.Path(
    "D:\DISUES PLANT\DoctorP_dataset"
    )

# Вывести количество изображений в датасете.
image_count = len(list(data_dir.glob('*/*.jpg')))

### CONFIG ###
BATCH_SIZE = 32
img_height = 128
img_width = 128

# 80 % для обучения
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=BATCH_SIZE)

# 20 % для проверки
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=BATCH_SIZE) 

# Атрибут для нахождения имен классов
class_names = train_ds.class_names

### ВИЗУАЛИЗАЦИЯ ДАННЫХ 
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show() 

# Нормализация данных
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Предварительная выборка с буферизацией
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


