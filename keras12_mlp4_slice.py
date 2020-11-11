
# 실습 train_test_splite 를 슬라이싱으로 바꿀것





#1. 데이터 #1부터 100
import numpy as np
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array([range(101, 201), range(711, 811), range(100)])

from sklearn.model_selection import train_test_split

# print(x[1][10])
# print(x.shape) #(3, 100)
# print(np.array(x))

#x.transpose()
x = np.transpose(x)
y = np.transpose(y)

print(x.shape) # (100, 3)


#데이터 슬라이싱
x_train = x[:70][0:]
y_train = y[:70][0:]
x_test = x[70:][0: ]
y_test = y[70:][0: ]






#행 무시 열 우선!!! (컬럼이 중요)
#모델을 만들때 행에 따라 만들어 지는게 아니라 컬럼 별로
# 특성 피쳐 컬럼 열 같은 말

# y= wx + b 
# y1, y2, y3 = w1x1 + w2x2 + w3x3 + b 

#모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3))# 컬럼이 3
model.add(Dense(5))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

loss = model.evaluate(x_test, y_test)

print("loss : ", loss)

y_predict = model.predict(x_test)
print("결과물 : ", y_predict)



#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict))


#R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)

#MLP 멀티 레이어 퍼센트론

# 실습 train_test_split을 행렬의 슬라이싱으로 바꿀 것


'''
#1. 데이터
import numpy as np

x=np.array((range(1,101), range(311, 411), range(100)))
y=np.array((range(101,201), range(711,811), range(100)))


print(x.shape) #(3, 100)
print(y.shape) #(3, 100)

x=np.transpose(x)
y=np.transpose(y)
# x.transpose()로도 가능

print(x.shape) #(100, 3)
print(y.shape) #(100, 3)

#슬라이싱
x_train = x[:100]
y_train = y[:100]

x_test = x[:90]
y_test = y[:90]

x_val = x[:70]
y_val = y[:70]



#------------- 여기 위에까지 데이터 구축 완성



#------------- 여기 아래서부터 모델 구성
# y1. y2, y3 = w1x1 + w2x2 + w3x3 +b 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5000))
model.add(Dense(1000))
model.add(Dense(9000))
model.add(Dense(3)) #출력 컬럼 3개


#---------------------나머지 완성
#3. 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#훈련 
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
print("결과물 : ", y_predict)

#RMSE 함수 사용자 정의
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE :", RMSE(y_test,y_predict))

#R2 함수
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 : ",r2)

print("x_test : ", x_test)
'''