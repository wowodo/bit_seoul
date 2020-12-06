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


x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test  = x_test.reshape(10000, 28*28).astype('float32')/255. 

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


def build_model(drop=0.5,
                optimizer="adam", 
                learning_rete=0.001, 
                node_value=64, 
                layer_num=1):

    inputs = Input(shape=(28*28,),name="input")
    x = Dense(512, activation="relu",name="hidden1")(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation="relu",name="hidden2")(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation="relu",name="hidden3")(x)
    x = Dropout(drop)(x)    

    outputs = Dense(10,activation="softmax",name="output")(x)
    model1 = Model(inputs=inputs, outputs=outputs)
    model1.compile(optimizer=optimizer(lr=learning_rete), metrics=["acc"],
                  loss="categorical_crossentropy")


    return model1
#위에 256 128 변수로 만들어서 아래 쓸수 있다 
#레이어는 변수명 잡아서 아래에다 넣을 수 있다
# optimizers = ['rmsprop', 'adam', 'adadelta'] adam 말고 2개가 더 늘었는데 더 안으로 들어가면 그 외에 것들

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    # optimizers = ['rmsprop', 'adam', 'adadelta']
    optimizers = [Adam, RMSprop]
    learning_rete=[0.001, 0.01]
    dropout = [0.1, 0.5]
    epochs = [20]
    node_value =[10, 20]
    layer_num = [1, 2]
    

    return{"batch_size":batches, 
          "optimizer":optimizers, 
          "learning_rete":learning_rete, 
          "drop":dropout, 
          "epochs":epochs, 
          "node_value":node_value, 
          "layer_num":layer_num }
       
hyperparameters = create_hyperparameter()



# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

wrapper_model = KerasClassifier(build_fn=build_model, verbose=2)
# keras 모델을 sk_learn에서 사용하기 위해 형변환

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(build_model, hyperparameters, cv=3) # fit에 문제가 생긴다
# search = GridSearchCV(wrapper_model, hyperparameters, cv=3) # wrapper를 씌워 사이킷런으로 가져온다
search = RandomizedSearchCV(wrapper_model, hyperparameters, cv=3) # wrapper를 씌워 사이킷런으로 가져온다

search.fit(x_train, y_train)

print(search.best_params_) # 최적의 파라미터를 찾는다.

acc = search.score(x_test, y_test)
print("최종 스코어 : " , acc)

#얼리 스타핑 
#epoch 디폴트값
'''
250/250 - 0s - loss: 0.1476 - acc: 0.9551
최종 스코어 :  0.9550999999046326

334/334 - 0s - loss: 0.1012 - acc: 0.9822
최종 스코어 :  0.982200026512146
'''


