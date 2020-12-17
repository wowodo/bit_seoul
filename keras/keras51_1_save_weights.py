from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
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



model.summary()

# model.save('./save/model_test01_1.h5')

#3,컴파일 훈련
modelpath ='./model/mnist-{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
ep = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
        save_best_only=True, mode ='auto')

# # 텐서 보드 
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
#                     write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
                #10개를 합친 값은 1이 되고 거기서 제일 큰걸
hist = model.fit(x_train, y_train, epochs=30, batch_size=512, verbose=1, validation_split=0.2, callbacks=[ep,cp])




##### model과 가중치 저장 바뀌지 않는 결과fit 이후에 세이브 하면 값이 저장된다####
model.save('./save/model_test02_2.h5')

#가중치만 저장 하는 놈
model.save_weights('./save/weight_test02.h5')

#4.평가 예측
result = model.evaluate(x_test, y_test, batch_size=512)
print("loss : ", result[0])
print("acc : ", result[1])

# #시각화
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6)) #단위 무엇인지 찾아볼것!

# plt.subplot(2,1,1) #(2행 1열에서 1번째 그림)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  # x값과, y값이 들어가야 합니다
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') #우측 상단에 legend(label 2개 loss랑 val_loss) 표시

# plt.subplot(2,1,2) #(2행 1열에서 2번째 그림)
# plt.plot(hist.history['accuracy'], marker='.', c='red')
# plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
# plt.grid()

# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()


# y_predict = model.predict(x_predict)
# # print(y_predict.shape) (10, 10)
# y_predict = np.argmax(y_predict, axis=1)
# # print(y_predict.shape) (10,)
# y_answer = np.argmax(y_answer, axis=1)

# print("예측값:" , y_predict)
# print("실제값: ", y_answer)

'''
결과값
loss :  1.201989769935608
acc :  0.6243000030517578

결과값 2
loss :  1.1969302892684937
acc :  0.6304000020027161
'''
