import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성(하이퍼 파라미터 튜닝)
#중간레이어는 히든 레이어라고 한다
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(350))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
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
model.fit(x_train, y_train, epochs=100)

#3,평가, 예측 
# loss, acc = model.evaluate(x, y, batch_size=1)
#loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_train, y_train)

print("loss : ", loss)
#print("acc : ", acc)

y_pred = model.predict(x_pred)
print("결과물 : /n : ", y_pred)

#실습 : 결과물 오차 수정. 미세조정