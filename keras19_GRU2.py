import numpy as np

x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input = np.array([50,60,70])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(13, 3, 1)

print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,GRU

model = Sequential()
model.add(GRU(30, input_length=(3), input_dim=(1)))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

# 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

x_input = np.array([50,60,70]) #(3,) -> (1, 3, 1)

x_input = x_input.reshape(1,3,1)

result = model.predict(x_input)

loss = model.evaluate(x_input, np.array([80]), batch_size=7)
print("loss :", loss)

print("loss : ", loss)
print(result)

#실습 LSTM  dhkstjdgktldh
#예측값 80