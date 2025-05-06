import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_augmentation import create_data_augmentation_layer

# Загрузка набора данных
data_dir = pathlib.Path("D:\DISUES PLANT\DoctorP_dataset")

# Конфигурация
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
AUTOTUNE = tf.data.AUTOTUNE

# Создание наборов данных
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True, 
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Получение классов
class_names = train_ds.class_names
num_classes = len(class_names)

# Предобработка (только one-hot кодирование)
def preprocess(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

# Визуализация аугментации перед обучением
def visualize_augmentation(dataset, augmentation_layer, num_images=9):
    plt.figure(figsize=(15, 8))
    for images, _ in dataset.take(1):
        augmented_images = augmentation_layer(images)
        for i in range(min(num_images, len(images))):
            # Оригинальное изображение
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title("Original")
            plt.axis("off")
            
            # Аугментированное изображение
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(augmented_images[i].numpy().astype("uint8"))
            plt.title("Augmented")
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Получить исходные метки (до one-hot)
labels_list = []
for images, labels in train_ds.unbatch():
    labels_list.append(labels.numpy())

# Преобразовать в массив NumPy
labels_array = np.array(labels_list)

# Вычислить веса с явным указанием параметров
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_array),
    y=labels_array
)
class_weights = dict(enumerate(class_weights))

# Создаем слой аугментации для проверки
augmentation_layer = create_data_augmentation_layer()
visualize_augmentation(train_ds, augmentation_layer)


train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# Оптимизация данных
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Создание модели с аугментацией
base_model = keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Разморозить последние 20 слоев
    layer.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = create_data_augmentation_layer()(inputs)  # Аугментация на лету
x = layers.Rescaling(1./255)(x)               # Нормализация после аугментации
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)  # Новый слой
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Компиляция и обучение
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

# Точная настройка
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

model.save('doctorp_resnet_model.h5')