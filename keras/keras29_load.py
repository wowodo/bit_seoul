# 모델 load 해서 완성!!

import numpy as np

dataset = np.array(range(1,101))
size = 5

# Dense 모델을 구성하시오.
# fit 까지만 할 것
# predic 까지

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)
print("=================")
#print(datasets)

x = datasets[:, 0:4]
y = datasets[:, 4]

print(x.shape) # (96, 4)
print(y.shape) # (96,)

x = np.reshape(x, (x.shape[0], x.shape[1], 1)) #3차원
print(x.shape) # (96, 4, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)

# LSTM 함수형 모델 구성
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.layers import Dense, Input

model = load_model('./save/keras28.h5')
model.add(Dense(5, name='king1'))
model.add(Dense(1, name='king2'))

model.summary()

'''
ValueError: All layers added to a Sequential model should have unique names.
 Name "dense" is already the name of a layer in this model. 
 Update the `name` argument to pass a unique name.
TypeError: add() got an unexpected keyword argument 'name'
NameError: name 'model50' is not defined
'''

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 예측, 평가
x_predict = np.array([97,98,99,100])
y_predict = model.predict(x_predict)
print("y_predict :", y_predict)

loss, mse = model.evaluate(x_test, y_test)
print("loss, mse :", loss, mse)

'''
TypeError:
'builtin_function_or_method' object is not subscriptable
ValueError: 
Input 0 of layer sequential is incompatible with the layer: 
expected ndim=3, found ndim=2. Full shape received: [None, 1]
'''