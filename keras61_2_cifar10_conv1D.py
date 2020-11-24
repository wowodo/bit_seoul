from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Conv1D
from tensorflow.keras.layers import Flatten, MaxPooling2D, MaxPooling1D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3), (10000, 32, 32,3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

#데이터 전처리 1.OneHotEncoding 라벨링 한다 y
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)

#fit 한 결과로  trainsform
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#스케일링 해준것
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*3, x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*3,x_test.shape[2]).astype('float32')/255.
# x_predict = x_predict.astype('float32')/255.

x_predict = x_train[:10]
y_answer = y_train[:10]

print(x_train.shape, y_train.shape)
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

#모델 구성
model = Sequential()
model.add(Conv1D(100, 3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Conv1D(55, 3, activation='relu'))
model.add(Conv1D(60, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2)) #이미지 자를때 중복은 없고 최대값을 가져간
model.add(Flatten()) #덴스 레이어로 
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()


#3,컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=2, mode='auto')

# # 텐서 보드 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
                    write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
                #10개를 합친 값은 1이 되고 거기서 제일 큰걸
history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

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
