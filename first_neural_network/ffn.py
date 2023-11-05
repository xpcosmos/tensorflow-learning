# Building with Keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Numpy
print(x_train.shape, y_train.shape)

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
plt.show()

# Method 1
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Converte um array de (28, 28) em um Array (784, 1)
    keras.layers.Dense(128, activation='relu'), # Camada escondida com interconectada função de ativação ReLu
    keras.layers.Dense(10) # Camada de output, igual numero de classes de predição
])

print(model.summary())

# Method 2

model = keras.Sequential
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation='relu'))
mo