from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape)#(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)#(50000, 1) (10000, 1)

# plt.imshow(x_train[0])
# plt.show()

#데이터 전처리 1.OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#스케일링 해준것
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_predict = x_test[:10]
y_answer = y_test[:10]
 
model = Sequential()
model.add(Conv2D(10, (3,3), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(360, (2,2),padding='valid'))
model.add(Conv2D(250, (3,3)))
model.add(Conv2D(150,(2,2),strides=2))
model.add(MaxPooling2D(pool_size=2)) #이미지 자를때 중복은 없고 최대값을 가져간
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

#. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping =EarlyStopping(monitor='loss', patience=2, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1, callbacks=[early_stopping])

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", accuracy)

y_predict = model.predict([x_predict])
print("예측값:" , y_predict)
y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("예측값:" , y_predict)
print("실제값: ", y_answer)

'''
ss: 0.4447 - accuracy: 0.9008
loss :  0.44470536708831787
acc :  0.9007999897003174
예측값: [9 0 0 3 0 2 7 2 5 5]
실제값:  [9 0 0 3 0 2 7 2 5 5]

'''