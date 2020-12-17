from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3), (10000, 32, 32,3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

#데이터 전처리 1.OneHotEncoding 라벨링 한다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_answer = y_train[:10]

#스케일링 해준것
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255
x_predict = x_predict.astype('float32')/255.

# from tensorflow.keras.models import load_model
# model = load_model('./save/model_test02_2.h5')

# #3,컴파일 훈련


# 체크 포인트는 모델과 가중치 모두 세이브 된다
from tensorflow.keras.models import load_model
model = load_model('./model/mnist-03-1.1026.hdf5')

#4.평가 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])


'''
결과값

loss :  1.2019896507263184
acc :  0.6243000030517578

결과값2
loss :  1.1969304084777832
acc :  0.6304000020027161

체크포인트
loss :  1.0898572206497192
acc :  0.6297000050544739
'''
