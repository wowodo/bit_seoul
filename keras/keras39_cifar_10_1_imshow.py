from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print("y_train : ",y_train[0])

print(x_train.shape, x_test.shape)
#(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)
#(50000, 1) (10000, 1)

plt.imshow(x_train[0])
plt.show()