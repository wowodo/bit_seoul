import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data() # 라벨이 없다.

x_train = x_train.reshape(x_train.shape[0],28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28*28)/255.

print(x_train[0])
print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size,input_shape=(784,),activation="relu"))
    model.add(Dense(units=784,activation="sigmoid"))
    return comfit(model)

def comfit(model):
    model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["acc"]) 
    model.fit(x_train, x_train, epochs=10)
    output = model.predict(x_test)
    return output

output_01 = autoencoder(hidden_layer_size=1)
output_02 = autoencoder(hidden_layer_size=2)
output_04 = autoencoder(hidden_layer_size=4)
output_08 = autoencoder(hidden_layer_size=8)
output_16 = autoencoder(hidden_layer_size=16)
output_32 = autoencoder(hidden_layer_size=32)

import matplotlib.pyplot as plt
import random
fig, axes = plt.subplots(7,5,figsize=(15,15))

random_imgs = random.sample(range(output_01.shape[0]),5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28),cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
