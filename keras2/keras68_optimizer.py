import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

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

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
#통계S  G그라이던트 d 가장 기본적인
# optimizer = Adam(lr=0.001) # 변수 만들고
# optimizer = Adadelta(lr=0.001)
# optimizer = Adagrad(lr=0.001)
# optimizer = Adamax(lr=0.001)
# optimizer = RMSprop(lr=0.001)
optimizer = SGD(lr=0.001)
# optimizer = Nadam(lr=0.001)


model.compile(loss='mse', optimizer=optimizer,
              metrics=['mse'])
    
model.fit(x, y, epochs=100, batch_size=1)
#3,평가, 예측 
# loss, acc = model.evaluate(x, y, batch_size=1)
loss, mse = model.evaluate(x, y, batch_size=1)

#x 값을 예측 0.999928483 머신은 다르다고 생각한다
y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)

'''
optimizer = Adam(lr=0.001)
loss :  0.007341448217630386 결과물 :  [[11.185092]]
----
optimizer = Adam(lr=0.01)
loss :  1.9986146418404793e-11 결과물 :  [[11.000007]]
----
optimizer = Adam(lr=0.1) # 변수 만들고
loss :  1.514450360673436e-07 결과물 :  [[10.999759]]

----------------------------------------------------

optimizer = Adadelta(lr=0.001)
loss :  0.00043644808465614915 결과물 :  [[10.974694]]
----
optimizer = Adadelta(lr=0.01)
loss :  0.0003152608696836978 결과물 :  [[10.975116]]
----
optimizer = Adadelta(lr=0.1)
loss :  0.004552842117846012 결과물 :  [[10.884586]]
-------------------------------------------------------

optimizer = Adagrad(lr=0.001)
loss :  1.3666278846358182e-06 결과물 :  [[10.998106]]
----
optimizer = Adagrad(lr=0.01)
loss :  2.2170748707139865e-05 결과물 :  [[10.999465]]
----
optimizer = Adagrad(lr=0.1)
loss :  77400.375 결과물 :  [[347.58475]]
-------------------------------------------------------

optimizer = Adamax(lr=0.001)
loss :  5.885005884920247e-05 결과물 :  [[11.006218]]
-----
optimizer = Adamax(lr=0.01)
loss :  2.7761281671701e-05 결과물 :  [[10.994843]]
-----
optimizer = Adamax(lr=0.1)
loss :  8105.41015625 결과물 :  [[-113.26932]]
-------------------------------------------------------
optimizer = RMSprop(lr=0.001)
loss :  0.007709974888712168 결과물 :  [[10.836005]]

--------------------------------------------------------


'''