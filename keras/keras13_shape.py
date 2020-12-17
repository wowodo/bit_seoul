#1. 데이터 #1부터 100
import numpy as np
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array([range(101, 201), range(711, 811), range(100)])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1, test_size=0.8, shuffle=False)

print(x[1][10])
print(x.shape) #(3, 100)
print(np.array(x))

#x.transpose()
x = np.transpose(x)
y = np.transpose(y)

print(x.shape) # (100, 3)

#행 무시 열 우선!!! (컬럼이 중요)
#모델을 만들때 행에 따라 만들어 지는게 아니라 컬럼 별로
# 특성 피쳐 컬럼 열 같은 말

# y= wx + b 
# y1, y2, y3 = w1x1 + w2x2 + w3x3 + b 

#모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(10, input_dim=3))
model.add(Dense(10, input_shape=(3, )))# 컬럼이 3
#(100,10,3) : input_shape=(10, 3) 행은 무시된다 (제일 앞에꺼 버린다)
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

model.fit(x_train, y_train, epochs=100, validation_data=0.2)

loss = model.evaluate(x_train, y_train)

print("loss : ", loss)

y_predict = model.predict(x_test)
print("결과물 : ",y_predict)



#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

#R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)

#MLP 멀티 레이어 퍼센트론

