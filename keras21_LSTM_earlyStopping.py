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
dense1 = LSTM(30, activation='relu')(input1)
dense1_1 = Dense(50, activation='relu')(dense1)
dense2 = Dense(40, activation='relu')(dense1_1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
output1 = Dense(1)(dense4)#activation='linear'인 상태

#모델 정의
model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping  #조기 종료  for 문 API
#loss 최소값 10번까지 넘어가면 멈추겠다 ()
# Early_Stopping = EarlyStopping(monitor='loss', patience=100, mode='min')
Early_Stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') #
#1000번 훈련하는 도중 최소값보다 올
model.fit(x, y, epochs=100, batch_size=1, verbose=1,
                callbacks=[Early_Stopping])

#예측



x_input = x_input.reshape(1, 3, 1)
print(x_input.shape)
y_predict = model.predict(x_input)
print("predict :", y_predict)


model.summary()
