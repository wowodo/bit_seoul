#Reduce 감소 시키다


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

#분류 할때 마지막   one hot encoding 하면   softmax

#다중 불류 마지막 액티베이션은 소프트 맥스  y라벨링 

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(30, (2,2),padding='valid'))
model.add(Conv2D(40, (3,3)))
model.add(Conv2D(20,(2,2),strides=2))
model.add(MaxPooling2D(pool_size=2)) #이미지 자를때 중복은 없고 최대값을 가져간
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.summary()

#3,컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ModelCheckpoint, ReduceLROnPlateau  #Plateau 찾아 보기

es = early_stopping = EarlyStopping(monitor='_val_oss', patience=5, mode='min')
model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
                #10개를 합친 값은 1이 되고 거기서 제일 큰걸
#
ck = ModelCheckpoint('./model', seve_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, ck])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1) #세번동안 감속히키지 않으면 50프로 감축시키고 (그중에 얼리스타핑 들어간다 )
#4.평가 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", accuracy)


y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)
y_answer = np.argmax(y_answer, axis=1)

print("예측값:" , y_predict)
print("실제값: ", y_answer)
