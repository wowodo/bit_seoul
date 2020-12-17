from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print("y_train : ",y_train[0])

print(x_train.shape, x_test.shape)#(60000,28,28),(10000,28,28)
print(y_train.shape, y_test.shape)#(60000,)      (10000,)

print("y_train : ", y_train)#[9 0 0 ... 3 0 5]

# plt.imshow(x_train[0],'gray')
# plt.show()

#데이터 전처리 1.OneHotEncoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)
print("y_train : ", y_train[0])

x_predict = x_train[:10]
y_answer = y_train[:10]


#스케일링 해준것
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2]).astype('float32')/255. #형변환
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

print("x_train : ",x_train[0])


 
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(360, (2,2),padding='valid'))
model.add(Conv2D(250, (3,3)))
model.add(Conv2D(150,(2,2),strides=2))
model.add(MaxPooling2D(pool_size=2)) #이미지 자를때 중복은 없고 최대값을 가져간
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping =EarlyStopping(monitor='loss', patience=2, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30, batch_size=28, verbose=1, callbacks=[early_stopping])

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", accuracy)

y_predict = model.predict(x_predict)
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