import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성(하이퍼 파라미터 튜닝)
#중간레이어는 히든 레이어라고 한다
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(40))
model.add(Dense(7))
model.add(Dense(1))


#3.컴파일, 훈련
#loss 최소하게 mse 을  adam 으로 최적화 하겠다 metrics 눈으로 보기위한 평가지표는 acc로 하겠다 (엑트러시 (정확도를))
# 
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
#정제된 모델에게 주겠다 epochs 훈련시키겠다
# model.fit(x, y, epochs=1000, batch_size=1)
model.fit(x, y, epochs=250)
#3,평가, 예측 
# loss, acc = model.evaluate(x, y, batch_size=1)
#loss, acc = model.evaluate(x, y)
loss = model.evaluate(x, y)

print("loss : ", loss)
#print("acc : ", acc)

y_pred = model.predict(x_pred)
print("결과물 : /n : ", y_pred)

#실습 : 결과물 오차 수정. 미세조정