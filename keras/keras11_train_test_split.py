#1. 데이터  슬라이싱
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, v_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True) # 디폴트 셔플은 True 명시를 안해줘도 된다,


x_train = x[:71] # 70개
y_train = y[:71]
x_test = x[71:]  # 30개
y_test = y[71:]
print(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델 구성(하이퍼 파라미터 튜닝)
#중간레이어는 히든 레이어라고 한다
model = Sequential() # Sequential()  순차적으로 하겠다
#model.add(Dense(30, input_dim=1))
model.add(Dense(30, input_shape=(1, )))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(10))
model.add(Dense(1))


#3.컴파일, 훈련
#loss 최소하게 mse 을  adam 으로 최적화 하겠다 metrics 눈으로 보기위한 평가지표는 acc로 하겠다 (엑트러시 (정확도를))
# *metrics list(여러가지)  mse  , mae, acc 로 받아 드린다 
# loss :  [0.00018591847037896514, 0.011714816093444824, 0.10000000149011612]
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

#정제된 모델에게 주겠다 epochs 훈련시키겠다 
# validation_split=0.2 발리데이션 20프로 자른다
# 많이 데이터를 돌릴떈 val 이 좋다
model.fit(x_train, y_train, epochs=100, validation_split=0.2)
        #   validation_data=(x_val,y_val))

#3,평가, 예측 
# loss, acc = model.evaluate(x, y, batch_size=1)
#loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_train, y_train)

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
