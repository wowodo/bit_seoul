from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape)#(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)#(50000, 1) (10000, 1)

# plt.imshow(x_train[0])
# plt.show()

#데이터 전처리 1.OneHotEncoding 라벨링
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#스케일링 해준것
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_predict = x_test[:10]
y_answer = y_test[:10]
 
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
model.add(Dense(110, activation = 'relu'))
model.add(Dense(210, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(100, activation = 'softmax') )
model.summary()





# 3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = './model/cifar100-{epoch:02d}--{val_loss:.4f}.hdf5' #현재 모델 경로(study에 model폴더)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
        save_best_only=True, mode='auto')

#파일명 : epoch:02니깐 2자리 정수 - val_loss .4니깐 소수 4째자리 표기

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')


model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, 
            verbose=1, validation_split=0.5, callbacks=[es])

model.save_weights('./save/weight_cifar100_01.h5')

model.save('./save/model_cifar100_01.h5')

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", accuracy)

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

'''
loss :  4.7979655265808105
acc :  0.2563999891281128
'''