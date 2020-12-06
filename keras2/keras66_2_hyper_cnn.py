#OneHotEncodeing
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
#1~9까지 손글씨
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) #(60000, 28 ,28) #(10000, 28 ,28)
print(y_train.shape, y_test.shape) #(60000, )       #(10000, )
print(x_train[0])  #28 행 28열 0은 빈칸
print(y_train[0])

print("x_test : ",x_test)
# plt.imshow(x_train[0],'gray')
# plt.show()


#데이터 전처리 1. OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)
print("y_train : ",y_train[0])


x_predict = x_train[:10]
y_answer = y_train[:10]


#스케일링 해준것
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. #형변환
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
x_predict = x_predict.reshape(10, 28, 28, 1).astype('float32')/255.
print("x_train : ",x_train[0])



# 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Input
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, Softmax
from tensorflow.keras.activations import relu, elu, selu, softmax, sigmoid




#분류 할때 마지막   one hot encoding 하면   softmax

def build_model(drop = 0.5,
                optimizer="adam",
                learning_rete=0.001,
                node_value=64,
                layer_num=1):

    inputs = Input(shape=(28,28,1))
    x = Conv2D(10, (3,3), padding='same')(inputs)
    x = Conv2D(30, (2,2) )(x)
    x = Conv2D(40, (3,3))(x)
    x = Conv2D(20,(2,2),strides=2)(x)
    x = MaxPooling2D(pool_size=2)(x) #이미지 자를때 중복은 없고 최대값을 가져간
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name="outputs")(x)

    model = Model(inputs = inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers,
                metrics=['accuracy'])

      return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizer =[Adam, RMSprop, Adadelta]
    learning_rete=[0.001, 0.1]
    epochs =[20]
    node_value=[10,20]
    layer_num=[1,2]

    return{"batca_size": batches,
            "optimizer": optimizer,
            "learning_rete":learning_rete,
            "drop":dropout,
            "epochs":epochs,
            "node_value":node_value,
            "layer_unm":layer_num
            }


hyperparameters = create_hyperparameter()

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

wrappers_model = KerasClassifier(build_fn=build_model, verbose=2)
# keras 모델을 sk_learn에서 사용하기 위해 형변환

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model,hyperparameters,cv=3,verbose=2)
search = RandomizedSearchCV(wrappers_model,hyperparameters,cv=3,verbose=2)

search.fit(x_train, y_train)

print(search.best_params_) # 최적의 파라미터를 찾는다.

acc = search.score(x_test, y_test)
print("최종 스코어 : " , acc)


'''
{'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 10}
1000/1000 [==============================] - 1s 1ms/step - loss: 0.0699 - acc: 0.9764
최종 스코어: 0.9764000177383423
'''