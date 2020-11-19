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
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(LSTM(30, input_shape=(3, 1)))#1개씩 잘라서 작업
model.add(LSTM(30, input_length=(3), input_dim=(1))) 
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



