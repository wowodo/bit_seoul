# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train[0])
# print("y_train[0]:", y_train[0])
print("x_train.shape:", x_train.shape) # x_train.shape: (60000, 28, 28)
print("x_test.shape:", x_test.shape) # x_test.shape: (10000, 28, 28) 
print("y_test.shape:", y_test.shape) # y_test.shape: (10000,)




# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]


# train_test_split
# scaler
# reshape

# train_test_split
# 입력 데이터가 이미 train과 test로 나뉘어 있어서 별도로 나누지 않는다
# validation은 fit에서 vadidation_split으로 나눈다


# Scaler
# 선택은 아무거나, 최적이라 생각하는 주관적 판단
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
# print(x_train[0])



# CNN을 위한 reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print("reshape x:", x_train.shape, x_test.shape)





# 2.모델
model = Sequential()
model.add( Conv2D(32, (3,3), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])) )
model.add( Conv2D(16, (1,1), padding='valid') )
model.add(MaxPooling2D(pool_size=(2,2)))

model.add( Conv2D(32, (3,3), padding='same') )
model.add( Conv2D(16, (1,1), padding='valid') )
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(10, activation = 'softmax') )
model.summary()



# 3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy', #CNN은 웬만하면 categorical_crossentropy
    optimizer='adam',
    metrics=['accuracy'] # 이젠 accuracy를 관찰할만하다
    )

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    mode='auto',
    verbose=2)

from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
modelpath = './model/fashion-{epoch:02d}-{val_loss:.4f}.hdf5'
model_check_point = ModelCheckpoint(
    filepath=modelpath,
    monitor='val_loss',
    save_best_only=True,
    mode='auto')

hist = model.fit(
    x_train, y_train,
    epochs=100, 
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping,]) 


model.save_weights('./save/weight_fashion_01.h5')

#모델만 불러오기
model.save('./save/model_fashion_01.h5')


# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss: ", loss)
print("accuracy: ", accuracy)


y_predict = model.predict(x_test) # 평가 데이터 다시 넣어 예측값 만들기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print("y_predict:\n", y)
# print("y_test:\n", y_test)
# print("y_predict:\n", y_predict)


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) # 단위는 찾아보자

plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['accuracy'], marker='.', c='red')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()


print("keras49_ModelCheckPoint_2_fashion end")

'''
loss:  0.6994113326072693
accuracy:  0.9009000062942505
'''