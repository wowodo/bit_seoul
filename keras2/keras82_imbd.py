from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras.utils.np_utils import to_categorical
#소스를 완성하시오 embedding


(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train.shape, x_test.shape)#(8982,) (2246,)
print(y_train.shape, y_test.shape)#(8982,) (2246,)


#y의 카테고리 개수 출력
category = np.max(y_train) +1
print("카테고리 : ", category) #46

#y의 유니크한 값
y_bunpo = np.unique(y_train)
print(y_bunpo)
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

##one hot incoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten,Conv2D, Conv1D

model = Sequential()
model.add(Embedding(10000, 120))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss='category_crossentropy', optimizer='adam',metrics=['acc'])

model.fit(x_train, y_train, batch_size=100, epochs=20)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('\n 테스트 정확도 : %.4f' % (model.evaluate(x_test, y_test)[1]))



