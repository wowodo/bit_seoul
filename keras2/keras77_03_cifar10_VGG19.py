'''
실습
가장 좋은 놈 어떤건지 결과차 비교용
기본튠 + 전이학습 9개 모델 비교

9개의 전이학습 모델들은
Fltten()다음에는 모두 똑같은 레이어로
구성할것

'''
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, BatchNormalization,Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3), (10000, 32, 32,3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)

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
# model.summary()
vgg19 = VGG19(include_top=False, input_shape=(32,32,3))

vgg19.trainable=True

model = Sequential()
model.add(vgg19)
model.add(Flatten())

model.add(Dense(64,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(10,activation="softmax"))
model.summary()


#3,컴파일 훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
ep = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp = ModelCheckpoint('./model', save_weights_only=True, save_best_only=True, monitor='val_loss',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# # 텐서 보드 
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
#                     write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
                #10개를 합친 값은 1이 되고 거기서 제일 큰걸
hist = model.fit(x_train, 
                y_train, 
                epochs=30, 
                batch_size=32, 
                verbose=1, 
                validation_split=0.2, 
                callbacks=[ep,cp])

#모델+가중치 저장하는 놈
# model.save('./save/model_cifar10_01_1.h5')

#가중치만 저장 하는 놈
# model.save_weights('./save/weight_cifar10_01.h5')

#4.평가 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])


y_predict = model.predict(x_predict)
# print(y_predict.shape) (10, 10)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict.shape) (10,)
y_answer = np.argmax(y_answer, axis=1)

print("예측값:" , y_predict)
print("실제값: ", y_answer)

'''
loss :  1.269937515258789
acc :  0.7170000076293945
예측값: [6 9 9 4 1 1 4 7 0 5]
실제값:  [6 9 9 4 1 1 2 7 8 3]
'''