import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu, selu, elu, softmax, sigmoid
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, Softmax #대문자로 하면 괄호 치고 안에 ? 넣고 쓸 수 있음

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


#반드시 액티베이션을 반드시 쓰지 않아도 된다. 선택적
def build_model(drop=0.5, 
                optimizer="adam", 
                learning_rate=0.001, 
                node_value1=512,
                node_value2=256,
                node_value3=128,
                layer_num=1,
                Act=Activation('relu')):
                
    inputs = Input(shape=(28*28,),name="input")

    x = Dense(node_value1, activation="relu",name="hidden1")(inputs)
    x = Dropout(drop)(x) #사위 레이어 갯수를 드랍만큼 줄인다

    x = Dense(node_value2, name="hidden2")(x) #엑티베이션 기본값인 리니어기 때문에 그 값 그대로 그냥 아래로 던져준다
    x = Act('relu')(x) #엑티베이션 레이어
    x = Dropout(drop)(x)

    x = Dense(node_value3,name="hidden3")(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(drop)(x)    

    outputs = Dense(10,activation="softmax",name="output")(x)
    model1 = Model(inputs=inputs, outputs=outputs)
    model1.compile(optimizer=optimizer(learning_rate=learning_rate),  metrics=["acc"],
                  loss="categorical_crossentropy")

    
    return model1

#위에 256 128 변수로 만들어서 아래 쓸수 있다 
#레이어는 변수명 잡아서 아래에다 넣을 수 있다
# optimizers = ['rmsprop', 'adam', 'adadelta'] adam 말고 2개가 더 늘었는데 더 안으로 들어가면 그 외에 것들

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizers = [RMSprop, Adam, Adadelta]
    learning_rate =[0.001, 0.1, 0.0001]
    node_value1=[10, 20]
    node_value2=[30, 40]
    node_value3=[50, 60]
    layer_num=[1,2]
    Act = [ReLU,LeakyReLU]

    
    # dropout = np.linspace[0.1, 0.3, 0.4]
    

    return{"batch_size":batches,
           "optimizer":optimizers,
           "learning_rate":learning_rate,
           "node_value1":node_value1,
           "node_value2":node_value2,
           "node_value3":node_value3,
           "Act":Act
            }
        #    "drop":dropout}
hyperparameters = create_hyperparameter()



# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
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

#얼리 스타핑 
#epoch 디폴트값
'''
# RandomCV
334/334 - 0s - loss: 0.6107 - acc: 0.8631
최종 스코어 :  0.863099992275238

200/200 - 0s - loss: 0.6848 - acc: 0.8112
최종 스코어 :  0.8112000226974487
'''
