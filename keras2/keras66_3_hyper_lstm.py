import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


x_train = x_train.reshape(60000, 28,28).astype('float32')/255.
x_test  = x_test.reshape(10000, 28,28).astype('float32')/255. 

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout,LSTM


def build_model(drop=0.5, optimizer="adam"):
    inputs = Input(shape=(x_train.shape[0], x_train.shape[1]),name="input")
    x = LSTM(512,activation="relu",name="hidden1")(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation="relu",name="hidden2")(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation="relu",name="hidden3")(x)
    x = Dropout(drop)(x)    

    outputs = Dense(10,activation="softmax",name="output")(x)
    model1 = Model(inputs=inputs, outputs=outputs)
    model1.compile(optimizer=optimizer, metrics=["acc"],
                  loss="categorical_crossentropy")

    return model1

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]

    # optimizers = ['rmsprop', 'adam', 'adadelta']
    optimizers = ['adam']

    dropout = [0.1, 0.2, 0.3]
    return{"batch_size":batches, "optimizer":optimizers}
        #    "drop":dropout}
hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model = KerasClassifier(build_fn=build_model, verbose=2)
# keras 모델을 sk_learn에서 사용하기 위해 형변환

from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
search = GridSearchCV(model,hyperparameters,cv=3,verbose=2)
search.fit(x_train,y_train)

print(search.best_params_) # 최적의 파라미터를 찾는다.

#케라스와 사이킷런 을 엮는거 레터?

'''
1200/1200 - 20s - loss: 0.8731 - acc: 0.7243
{'batch_size': 50, 'optimizer': 'adam'}
'''

#러닝 메이트 loss에서 0을 찾아 가는 과정
#그떄 간격이 크면 0을 찾지 못한다 
#작게 하면 찾기는 좋지만 시간이 너무 오래 걸린다
#prameter로 조절할 수 있다 optimizer