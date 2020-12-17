#1. 데이터
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #(4, 3)
y = np.array([4,5,6,7])                         #(4, )  스칼라 4개

print("x.shape",x.shape)
print("y.shape",y.shape)


x = x.reshape(x.shape[0], x.shape[1], 1) #LSTM 3차원 쉐이프를 원한다 몇개씩 자를지
# x = x.reshape(4, 3, 1)
print("x.shape",x.shape)
#2. 모델 구성
# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()
model.add(SimpleRNN(30, activation='relu', input_shape=(3, 1))) #1개씩 잘라서 작업
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

x_input = np.array([5,6,7]) #(3,) -> (1, 3, 1)

x_input = x_input.reshape(1,3,1)

result = model.predict(x_input)

print(result)


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 30)                960
_________________________________________________________________
dense (Dense)                (None, 20)                620
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,801
Trainable params: 1,801
Non-trainable params: 0
_________________________________________________________________
'''
