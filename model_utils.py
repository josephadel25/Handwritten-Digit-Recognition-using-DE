import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
os.environ["TF_DETERMINISTIC_OPS"] = "1"
mixed_precision.set_global_policy('float32')

def build_model(seed=42):
    initializer = GlorotUniform(seed=seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    return model

def flatten_weights(model):
    flat_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for w in weights:
                flat_weights.extend(w.flatten())
    return np.array(flat_weights)

def set_model_weights(model, flat_weights):
    pointer = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            shapes = [w.shape for w in weights]
            params = []
            for shape in shapes:
                size = np.prod(shape)
                param = flat_weights[pointer:pointer+size].reshape(shape)
                params.append(param)
                pointer += size
            layer.set_weights(params)

def load_mnist_data(train_path='dataset/mnist_train.csv', test_path='dataset/mnist_test.csv'):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        x_train = train_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values / 255.0
        y_train = to_categorical(train_data.iloc[:, 0], num_classes=10)

        x_test = test_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values / 255.0
        y_test = to_categorical(test_data.iloc[:, 0], num_classes=10)

        x_train = x_train.reshape(-1, 784) 
        x_test = x_test.reshape(-1, 784)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        return x_train, y_train, x_val, y_val, x_test, y_test