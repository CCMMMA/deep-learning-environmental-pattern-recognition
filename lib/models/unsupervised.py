import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU


def AutoEncoder_OneLayer(encoding_dim, n_features):
    return Sequential([
        Dense(encoding_dim, input_shape=(n_features,), activation='relu', name="encoder"),
        Dense(n_features, name="decoder")
    ])
