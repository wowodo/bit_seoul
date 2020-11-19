# mnist2_DNN 실습
# (60000, 28, 28) -> (60000, ?) colum이 (60000,784)

'''
실습1. test데이터를 10개 가져와서 predict 만들 것
    - 원핫 인코딩을 원복할 것
    print('실제값 : ', 어쩌구 저쩌구) #결과 : [3 4 5 2 9 1 3 9 0]
    print('예측값 : ', 어쩌구 저쩌구) #결과 : [3 4 5 2 9 1 3 9 1]
실습2. 모델: es 적용, tensorboard도 넣을 것
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28), (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,), (10000,)

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000, 10), (10000, 10)

x_train = x_train.reshape.astype('float32')/255. #형변환
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255.

# x_predict=x_predict.reshape(10,28*28).astype('float32')/255.

x_predict = x_train[:10]
y_real = y_train[:10]

# .astype : 형변환

print(x_train.shape, x_test.shape, x_predict.shape)
#(60000, 28, 28, 1) (10000, 28, 28, 1) (10, 28, 28, 1)
#(60000, 28, 14, 2) => 데이터 총 갯수만 맞으면 됩니다!



# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(28*28,1)))
model.add(Dense(20, activation='relu')) 
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu')) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax')) #분류한 값의 총 합은 1



# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

es = EarlyStopping(monitor='loss', patience=50, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train,y_train, epochs=1, batch_size=32, 
                verbose=1, validation_split=0.2, callbacks=[es, to_hist])

# 4. 평가, 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)

# np.argmax 함수를 사용하면 인코딩된 데이터를 역으로 되돌릴 수 있다.
y_predict = model.predict(x_predict)

y_predict_recovery = np.argmax(y_predict, axis=1)
y_real = np.argmax(y_real, axis=1)

print('실제값 : ',y_real) #정답
print('예측값 : ',y_predict_recovery) #내가 써 낸 답안지


##LSTM으로 바꿔줘야 하는데 어떻게 하는지 모르겠다.