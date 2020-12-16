#

import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

print(x_train.shape)
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten

def autoencoder (hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(154, (2,2), strides=(1,1),
                    padding='valid', input_shape=(28,28,1)
                     ))
    model.add(Conv2D(128, padding='valid',kernel_size= (2,2) ) )
    model.add(Flatten())
    model.add(Dense(units=128))
    model.add(Dense(units=64))
    model.add(Dense(units=128))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train.reshape(60000,784), epochs=10, batch_size=512)
outputs = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize=(20,7))
random_images = random.sample(range(outputs.shape[0]), 5)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(outputs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("input", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(outputs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("output", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()
