from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train[0])
print("y_train : ",y_train[0])

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

plt.imshow(x_train[0])
plt.show()

#데이터 전처리 1.OneHotEncoding 라벨링 한다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_answer = y_train[:10]


#스케일링 해준것
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255
x_predict = x_predict.astype('float32')/255.


# 모델

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

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(30, (2,2),padding='valid'))
model.add(Conv2D(40, (3,3)))
model.add(Conv2D(20,(2,2),strides=2))
model.add(MaxPooling2D(pool_size=2)) #이미지 자를때 중복은 없고 최대값을 가져간
model.add(Flatten()) #덴스 레이어로 
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 10)        130
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 31, 30)        1230
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 29, 40)        10840
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 20)        3220
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 7, 7, 20)          0
_________________________________________________________________
flatten (Flatten)            (None, 980)               0
_________________________________________________________________
dense (Dense)                (None, 30)                29430
_________________________________________________________________
dense_1 (Dense)              (None, 10)                310
=================================================================
Total params: 45,160
Trainable params: 45,160
Non-trainable params: 0
_________________________________________________________________
'''

#3,컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=2, mode='auto')

# 텐서 보드 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
                    write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
                #10개를 합친 값은 1이 되고 거기서 제일 큰걸
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping,to_hist])

#4.평가 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", accuracy)

y_predict = model.predict(x_predict)
# print(y_predict.shape) (10, 10)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict.shape) (10,)
y_answer = np.argmax(y_answer, axis=1)

print("예측값:" , y_predict)
print("실제값: ", y_answer)
'''
val_accuracy: 0.5558
racy: 0.5530
loss :  1.2594857215881348
acc :  0.5529999732971191
예측값: [3 9 9 4 9 9 4 7 8 7]
실제값:  [6 9 9 4 1 1 2 7 8 3]
'''
