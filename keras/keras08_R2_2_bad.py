#실습
#R2 를 음수가 아닌 0.5 이하로 줄이기
# 레이어는 인풋과 아웃풋을 포함 7ㅐ 이상(히든 5개 이상)
#히든 레이어 노드는 레이어딩 각각 최소 10개 이상
# betch_size = 1
# epochs = 100  이상
#데이터 조작 금지



import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) #예측하고 싶은거

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성(하이퍼 파라미터 튜닝)
#중간레이어는 히든 레이어라고 한다
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))


#3.컴파일, 훈련
#loss 최소하게 mse 을  adam 으로 최적화 하겠다 metrics 눈으로 보기위한 평가지표는 acc로 하겠다 (엑트러시 (정확도를))
# *metrics list(여러가지)  mse  , mae, acc 로 받아 드린다 
# loss :  [0.00018591847037896514, 0.011714816093444824, 0.10000000149011612]
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

#정제된 모델에게 주겠다 epochs 훈련시키겠다 
# batch(일괄작업)_size=1 (하나씩 잘라서 ) 1,2,3,4,5 1000번 훈련
# model.fit(x, y, epochs=1000, batch_size=1)
model.fit(x_train, y_train, epochs=101, batch_size=1)

#3,평가, 예측 
# loss, acc = model.evaluate(x, y, batch_size=1)
#loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)
#print("acc : ", acc)

# 프레딕트 예측한 값과 평가한다
y_predict = model.predict(x_test)
print("결과물 : /n : ", y_predict)

#실습 : 결과물 오차 수정. 미세조정

# 사용자 정의로 만듬 
# sklearn
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

#R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)