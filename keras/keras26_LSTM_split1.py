import numpy as np


#1. 데이터
#train, test할 필요 없이 fit까지만
 
dataset = np.array(range(1,11))
size = 5

#데이터 전처리 
def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]

        #aaa.append 줄일 수 있음
        #소스는 간결할수록 좋다
        # aaa.append([item for item in subset])
        aaa.append(subset)
        
        
        
    # print(type(aaa))
    return np.array(aaa)


dataset = split_x(dataset, size)
# print(dataset)


#2차원 배열 slicing은 numpy 이용하기 

#[(행 범위), (열 범위)]
#범위의 경우, 0:n이라고 했을 때 index는 n-1까지만

#[][] 사용하려면 사용법이 기존과 다르다
#첫 번째 []에서 indexing 후 그 결과를 가지고 다음 []로 indexing
### 열 먼저 하고 행을 하는 게 낫지 않나?



x = dataset[:, :4]
y = dataset[0:6, 4]

x = x.reshape(6, 4, 1)


#모델을 구성하시오.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input #LSTM도 layer

input1 = Input(shape=(4, 1))
dense1_1 = LSTM(30, activation='relu', name='king1')(input1)
dense1_2 = Dense(10, activation='relu', name='king2')(dense1_1)
dense1_3 = Dense(4, activation='relu', name='king3')(dense1_2)
output1 = Dense(1, activation='linear')(dense1_3)

#모델 정의
model = Model(inputs =input1, outputs=output1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.7
)

#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y,callbacks=[early_stopping],epochs=1000, batch_size=1)

# 예상, 훈련
# 7 8 9 10 에 대한 predict 및 튜닝

x_predict = np.array([7, 8, 9, 10])
x_predict = x_predict.reshape(1, 4, 1)


y_predict = model.predict(x_predict)
print("예측값: ", y_predict)

loss,mse = model.evaluate(x_test, y_test)
print(loss, mse) 