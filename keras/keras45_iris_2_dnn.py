#다중 분류
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

dataset = load_iris()

x = dataset.data
y = dataset.target

print(x)
print(y)

print(x.shape, y.shape)#(150, 4) (150,)

# 데이터 전처리
# train, test 분리. validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7
)


#데이터 전처리 1.OneHotEncoding 라벨링 한다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_predict = x_train[:10]
y_answer = y_train[:10]


print("x_train : ", x_train)
#x는 스케일 y는  onehotEncoding
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

model = Sequential()
model.add(Dense(30, input_shape=(4, )))
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))
model.summary()

#3,컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

# 텐서 보드 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, 
                    write_graph=True, write_images=True)


model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping,to_hist])

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