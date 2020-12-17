import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성(하이퍼 파라미터 튜닝)
model = Sequential()
model.add(Dense(300, input_dim=1))
model.add(Dense(5000))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))


#3.컴파일, 훈련
#loss 최소하게 mse 을  adam 으로 최적화 하겠다 metrics 눈으로 보기위한 평가지표는 acc로 하겠다 (엑트러시 (정확도를))
model.compile(loss='mse', optimizer='adam',
              metrics=['acc'])
#정제된 모델에게 주겠다 epochs 훈련시키겠다 
# batch(일괄작업)_size=1 (하나씩 잘라서 ) 1,2,3,4,5 1000번 훈련
# model.fit(x, y, epochs=1000, batch_size=1)
model.fit(x, y, epochs=100, batch_size=1)
#3,평가, 예측 
# loss, acc = model.evaluate(x, y, batch_size=1)
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)




