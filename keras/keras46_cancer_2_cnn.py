#유방암 데이터 -> 걸렸는지 / 안 걸렸는지: 2진 분류
#Classes 2

#sigmoid(relu, softmax 이전에 나왔던 것) - MinMaxScaler와 유사
# 함수값이 (0, 1)로 제한된다.
# 중간 값은 1/2이다.
# 매우 큰 값을 가지면 함수값은 거의 1이며, 매우 작은 값을 가지면 거의 0이다.

#단점
# 1) Gradient Vanishing 현상:
# 미분함수에 대해 x=0에서 최대값 1/4 을 가지고, input값이 일정이상 올라가면 미분값이 거의 0에 수렴하게된다. 
# 이는 |x|값이 커질 수록 Gradient Backpropagation시 미분값이 소실될 가능성이 크다.
# 2) 함수값 중심이 0이 아니다. : 함수값 중심이 0이 아니라 학습이 느려질 수 있다. 
# 만약 모든 x값들이 같은 부호(ex. for all x is positive) 라고 가정하고 아래의 파라미터 w에 대한 미분함수식을 살펴보자. 
# ∂L∂w=∂L∂a∂a∂w 그리고 ∂a∂w=x이기 때문에, ∂L∂w=∂L∂ax 이다. 위 식에서 모든 x가 양수라면 결국 ∂L∂w는 ∂L∂a 부호에 의해 결정된다.
# 따라서 한 노드에 대해 모든 파라미터w의 미분값은 모두 같은 부호를 같게된다. 
# 따라서 같은 방향으로 update되는데 이러한 과정은 학습을 zigzag 형태로 만들어 느리게 만드는 원인이 된다.
# exp 함수 사용시 비용이 크다.








import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout



#1. 데이터 
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)




#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0], 30, 1, 1)
x_test = x_test.reshape(x_test.shape[0], 30, 1, 1)


# print(x_train.shape)


x_predict = x_train[30:40]
y_answer = y_train[30:40]


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(30, 1, 1))) #padding 주의!
# model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.5))


model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) #ouput



#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=1
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=1)

print("=======cancer_cnn=======")
model.summary()
print("loss: ", loss)
print("acc: ", accuracy)


#정답

#예측값
y_predict = model.predict(x_predict)
y_predict = y_predict.reshape(10,)

print("예측값: ", y_predict)
print("정답: ", y_answer)

'''
=======cancer_cnn=======
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 1, 32)         320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 1, 128)        36992
_________________________________________________________________
dropout (Dropout)            (None, 30, 1, 128)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 1, 64)         73792
_________________________________________________________________
dropout_1 (Dropout)          (None, 30, 1, 64)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 30, 1, 256)        147712
_________________________________________________________________
dropout_2 (Dropout)          (None, 30, 1, 256)        0
_________________________________________________________________
flatten (Flatten)            (None, 7680)              0
_________________________________________________________________
dense (Dense)                (None, 512)               3932672
_________________________________________________________________
=================================================================
Total params: 4,192,001
Trainable params: 4,192,001
Non-trainable params: 0
_________________________________________________________________
loss:  0.31286901235580444
acc:  0.9415204524993896
예측값:  [1.0000000e+00 9.9999928e-01 9.9999988e-01 5.3581798e-06 9.9999988e-01
 9.9998581e-01 3.6942086e-07 1.0000000e+00 8.8136122e-03 1.4481157e-13]
정답:  [1 1 1 0 1 1 0 1 0 0]
'''