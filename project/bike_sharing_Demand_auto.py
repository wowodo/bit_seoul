import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

x_train = np.load("./project/bike_x.npy")
x_test = np.load("./project/bike_x.npy")
y_train = np.load("./project/bike_y.npy")
y_test = np.load("./project/bike_y.npy")

print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

#Initialize the image classifier
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=3)
#feed the image classifier with training data
clf.fit(x_train, y_train, epochs=50)

# Predict with the best model
predicted_y = clf.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
# clf.summary()
