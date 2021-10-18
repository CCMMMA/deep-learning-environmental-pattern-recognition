import tensorflow as tf
from tensorflow.keras import layers


def DenseWeather(input_shape, num_classes):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    return tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
