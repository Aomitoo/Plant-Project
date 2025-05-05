import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

import pathlib

# Download dataset
data_dir = pathlib.Path(
    "D:\DISUES PLANT\DoctorP_dataset"
    )

# Вывести количество изображений в датасете.
image_count = len(list(data_dir.glob('*/*.jpg')))

if __name__ == "main":
    print(image_count)

