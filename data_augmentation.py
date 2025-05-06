import tensorflow as tf
from tensorflow.keras import layers

def create_data_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.125),  # 45 градусов (0.125 * 360)
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(height_factor=(-0.2, 0.2)),  # Растяжение/сжатие до ±10%
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.2),
    ])