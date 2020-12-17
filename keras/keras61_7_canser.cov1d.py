#DNN레이어 , 이진분류(sigmoid)
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Conv1D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(569,30,1)

print(x.shape) #569,30
print(y.shape) #569
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = Sequential()
model.add(Conv1D(10,2 , activation='relu', input_shape=(30,1)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='loss',patience=20, mode='auto')

modelPath = './save/cancer/{epoch:02d}keras49_cp_7_caner_dnn_bi--{val_loss:.4f}.hdf5'
checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto' )

hist=model.fit(x_train, y_train, epochs=300, batch_size=4, validation_split=0.2, callbacks=[ealystopping, checkPoint])

checkPoint = ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True, mode='auto' )

loss, acc=model.evaluate(x_test, y_test, batch_size=4)

x_predict = x_test[20:30]
y_answer = y_test[20:30]

path = "./save/cancer/modelSave"

y_predict = model.predict(x_predict)

# y_predict = np.argmax(y_predict, axis=1)
# y_answer = np.argmax(y_answer, axis=1)

print("acc",acc)
print("loss",loss)
print("정답",y_answer)
print("예상값",y_predict)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6) ) # 단위 찾아보기

plt.subplot(2, 1, 1) #2행 1열의 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss' )
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss' )
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2, 1, 2) #2행 2열의 첫 번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])
plt.show()


'''
'''