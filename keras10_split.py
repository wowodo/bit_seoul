#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:71] # 70개
y_train = y[:71]
x_test = x[71:]  # 30개
y_test = y[71:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics='mae')

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

loss = model.evaluate(x_train, y_train)

print("loss : ", loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_train)
print("R2 : ", r2)