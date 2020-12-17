#실습  return_sequences튠


import numpy as np

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], 
             [9, 10, 11], [10, 11 ,12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])


y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
print(y.shape)
x_input = np.array([50, 60, 70])


# print(x.shape)

#shape 맞추기
# x = x.reshape(13, 3, 1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#실습: LSTM 완성
#예측값: 80

model = Sequential()
model.add(Dense(200, input_dim=3))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


#과제: 80 최근접치
#(3, 1) input_length=3 input_dim=1

#LSTM layer를 둘 다 사용할 때
#오류
#ValueError: 
# Input 0 of layer lstm_1 is incompatible with the layer: 
# expected ndim=3, found ndim=2. Full shape received: [None, 30]
# input과 output의 shape가 맞지 않을 때 나타나는 오류
#3차원을 기대 했는데 너가 2차원밖에 안보냈다

#3차원으로 바꿔주는 작업니 필요하다 return_sequences=True 차원 그대로 보낸다 (None, 3, 30)  
# model.add(LSTM(30, input_shape=(3, 1), return_sequences=True))
# model.add(LSTM(40, activation='relu'))
# # model.add(Dense(70, activation='relu'))
# # model.add(Dense(100, activation='relu'))
# # model.add(Dense(50, activation='relu'))
# # model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


#early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') #min/max 헷갈릴 때
model.fit(x, y, epochs=500, batch_size=1, callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("\nloss: ", loss)
#x_input reshape
# dense 레이어 맞게 인풋쉐이프 맞춰준다.
x_input = x_input.reshape(1, 3) 
y_predict = model.predict(x_input)

print(y_predict)

