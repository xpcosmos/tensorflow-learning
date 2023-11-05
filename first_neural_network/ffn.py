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
    keras.layers.Flatten(input_shape=(28,28)), # Convert a array (28,28) to (784,1)
    keras.layers.Dense(128, activation='relu'), # Hidden layer interconected and with ReLu Activation function
    keras.layers.Dense(10) # Output layer. Number of neurons is the same as the number of categories 
])

print(model.summary())

# Method 2

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28,28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10))

# print(model.summary())

# from_logits is true because the NN doesn't have SoftMax layer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001) # New Adam runs slowly in Mac M1/M2
metrics = ['accuracy']

model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

# Number of samples processed before the model is updated
BATCH_SIZE = 64
# Number of complete passes through the training dataset
EPOCHS = 5

# Fitting model to the train data
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=1) 
