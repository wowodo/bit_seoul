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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.layers import Dropout


#분류 할때 마지막   one hot encoding 하면   softmax

#다중 불류 마지막 액티베이션은 소프트 맥스  y라벨링 
# model = Sequential()
# model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
# model.add(Conv2D(20, (2,2),padding='valid'))
# model.add(Conv2D(30, (3,3)))
# model.add(Conv2D(40,(2,2),strides=2))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='softmax'))


#과적합이 잡힌다 드랍아웃을 쓰면
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Dropout(0.2))# 100개의 로드가 있다면 80개만 쓰겠다
model.add(Conv2D(30, (2,2),padding='valid'))
model.add(Dropout(0.2))
model.add(Conv2D(40, (3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(20,(2,2),strides=2))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2)) #이미지 자를때 중복은 없고 최대값을 가져간
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

'''
드랍아웃 하고도 값을 가지고 있는데  
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 10)        50
_________________________________________________________________
dropout (Dropout)            (None, 28, 28, 10)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 27, 27, 30)        1230
_________________________________________________________________
dropout_1 (Dropout)          (None, 27, 27, 30)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 25, 40)        10840
_________________________________________________________________
dropout_2 (Dropout)          (None, 25, 25, 40)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 20)        3220
_________________________________________________________________
dropout_3 (Dropout)          (None, 12, 12, 20)        0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 6, 6, 20)          0
_________________________________________________________________
dropout_4 (Dropout)          (None, 6, 6, 20)          0
_________________________________________________________________
flatten (Flatten)            (None, 720)               0
_________________________________________________________________
dense (Dense)                (None, 100)               72100
_________________________________________________________________
dropout_5 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010
_________________________________________________________________
dropout_6 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
=================================================================
Total params: 88,560
Trainable params: 88,560
Non-trainable params: 0
_____________________________
'''

model.summary()

#3,컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
                #10개를 합친 값은 1이 되고 거기서 제일 큰걸
history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4.평가 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", accuracy)


y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("예측값:" , y_predict)
print("실제값: ", y_answer)


'''
실습 1. test 데이터 10개를 가져와서 predict  만들것
            -원 핫 인코딜을 원복할것
            pint("실제값 :", 어쩌구 저쩌구) #결과 [3,4,5,2,9,1,3,9,0]
            pint("예측값 :", 어쩌구 저쩌구) #결과 [3,4,5,2,9,1,3,9,1]
실습 2. 모델: es, tensorboard

'''
'''
드랍아웃 후 (무족건 잘된다고 볼수 없다 레이어마다 넣거나 특정레이어에 넣어야 한다)
racy: 0.9571
loss :  0.15764513611793518
acc :  0.957099974155426
예측값: [5 0 4 1 9 2 1 3 1 4]
실제값:  [5 0 4 1 9 2 1 3 1 4]
'''



'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 10)        50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 27, 27, 20)        820
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 25, 30)        5430
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 40)        4840
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 6, 6, 40)          0
_________________________________________________________________
flatten (Flatten)            (None, 1440)              0
_________________________________________________________________
dense (Dense)                (None, 100)               144100
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010
=================================================================
Total params: 156,250
Trainable params: 156,250
Non-trainable params: 0
_________________________________________________________________
'''

#cnn,dnn 액티베이션
#Scaler


