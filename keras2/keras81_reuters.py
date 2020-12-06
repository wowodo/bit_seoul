from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib as plt

(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train.shape, x_test.shape)#(8982,) (2246,)
print(y_train.shape, y_test.shape)#(8982,) (2246,)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))#87
print(len(x_train[11]))#59

#y의 카테고리 개수 출력
category = np.max(y_train) +1
print("카테고리 :", category) #카테고리 : 46 문자으로 되있고

#y의 유니크한 값
y_bunpo = np.unique(y_train)
print(y_bunpo) 
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
'''

