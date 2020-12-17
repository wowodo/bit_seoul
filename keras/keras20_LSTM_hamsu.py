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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input,LSTM

input1 = Input(shape=(3,1))
dense1 = LSTM(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, actication='relu')(dense2)
output1 = Dense(1)(dense3)#activation='linear'인 상태

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1, verbose=1)

y_predict = model.predict(x_input)
print("predict :", y_predict)


model.summary()

# # 실행
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=200, batch_size=1)

# x_input = np.array([50,60,70]) #(3,) -> (1, 3, 1)

# x_input = x_input.reshape(1,3,1)

# result = model.predict(x_input)

# loss = model.evaluate(x_input, np.array([80]), batch_size=7)
# print("loss :", loss)

# print("loss : ", loss)
# print(result)

# #실습 LSTM  dhkstjdgktldh
#예측값 80