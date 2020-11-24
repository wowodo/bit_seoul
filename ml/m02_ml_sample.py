#꽃잎과 줄기를 보고 어떤 꽃인지 판별하는 데이터, 다중분류
#x column=4 y label:1

import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM


from sklearn.svm import LinearSVC #사이키런 모델 선형으로


#################원핫 인코딩을 안해돋 됨 모델도 만들어져 있다

#1. 데이터 

#데이터 구조 확인

x, y = load_iris(return_X_y=True) #data(X)와 target(Y)으로 구분되어 있다
# x = dataset.data
# y = dataset.target


# print(x.shape) #(150, 4)
# print(y.shape) #(150,)

# #OneHotEncoding (다중분류)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]).astype('float32')/255.
# x_test = x_test.reshape(x_test.shape[0], x_train.shape[1]).astype('float32')/255.

print(x_train.shape)

# #OneHotEncoding (다중분류) 주석 처리
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# x_predict = x_train[:10]
# y_answer = y_train[:10]

#2. 모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer

# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
#                                         #Conv2d의 activatio n default='relu'
#                                         #LSTM의 activation default='tanh'
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
#                                             #즉 softmax를 사용하려면 OneHotEncoding 해야

model = LinearSVC()

# #3. 컴파일 및 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(
#     x_train,
#     y_train,
#     validation_split=0.2,
#     epochs=1000, batch_size=32
# )

model.fit(x_train, y_train)



#4. 평가, 예측
# loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

# print("=======iris_dnn=======")
# model.summary()
# print("loss: ", loss)
# print("acc: ", accuracy)

result = model.score(x_test, y_test)
print("score : ",result)

y_predict = model.predict(x_test)


# #정답
# y_answer = np.argmax(y_answer, axis=1)

# #예측값
# y_predict = model.predict(x_predict)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_answer)

