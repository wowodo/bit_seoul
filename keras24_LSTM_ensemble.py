import numpy as np
from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

#1.데이터
x1 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], 
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])

x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2, 3, 4], [3, 4, 5],[4, 5, 6]])

y1 = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = array([55, 65, 75]) #(3,) 
x2_predict = array([65,75,85])

x1 = x1.reshape(13, 3, 1)
x2 = x2.reshape(13, 3, 1)

x1_predict = x1_predict.reshape(1, 3, 1) #하나의 데이터에 3개의 요소가 있고 하나씩 자르겠다 
x2_predict = x2_predict.reshape(1, 3, 1)

print(x1.shape)
print(x2.shape)
print(y1.shape)

##############실습 : 앙상블 모델을 만드시오
# from sklearn.model_selection import train_test_split
# x1_train, x_test, x2_train, x_test, = train_test_split(
#     x1,x2, shuffle=True, train_size=0.7
# )
# from sklearn.model_selection import train_test_split
# y1_train, y1_test = train_test_split(
#     y1, shuffle=True, train_size=0.7
# )

#모델1
input1 = Input(shape=(3, 1))
dense1_1 = LSTM(30, activation='relu', name='king1')(input1)
dense1_2 = Dense(10, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(4, activation='relu', name='king3')(dense1_2)
output1 = Dense(1, activation='linear')(dense1_3)

#모델2
input2 = Input(shape=(3, 1))
dense2_1 = LSTM(30, activation='relu', name='qeen1')(input2)
dense2_2 = Dense(10, activation='relu', name='qeen2')(dense2_1)
dense2_3 = Dense(6, activation='relu', name='qeen3')(dense2_2)
output2 = Dense(1, activation='linear')(dense2_3)



from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2])

middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(10)(middle1)

output1 =Dense(30)(middle1)
output1 =Dense(7)(output1)
output1 =Dense(1)(output1)


#모델 정의
model = Model(inputs = [input1,input2],
        outputs = output1)

model.summary()

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
# 훈련
model.fit(
    [x1, x2], y1, epochs=100, batch_size=1, verbose=1)

#4.평가,예측

# result = model.evaluate([x1, x2], y1_test,
#                         batch_size=1)

y1_predict = model.predict([x1_predict, x2_predict])

y2_predict = model.predict([x2_predict, x1_predict])
print(y2_predict)
print("결과물 : ",y1_predict)
