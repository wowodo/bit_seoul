from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import datetime
from sklearn.metrics import r2_score
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
#컨브1디를 쓰려면 2차원에서 
x = np.load('./npy/all_scale_x_11025sr.npy') 
y = np.load('./npy/all_scale_y_11025sr.npy') 
y = y-48
#  x_predict = np.load('./npy/mag_tmp.npy')

print(x.shape) #(6084, 37)
print(y.shape) #(6084,)


x = x.reshape(x.shape[0],x.shape[1],1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x_train.shape, x_test.shape)#(4867, 37, 1) (1217, 37, 1)
print(y_train.shape, y_test.shape)#(4867,) (1217,)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1).astype('float32')/255.
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1).astype('float32')/255.


#모델링

model = Sequential()
model.add(Conv1D(256, 3, activation='relu', padding='same', input_shape=(x_train.shape[1],1)))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Conv1D(512, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(37, activation='softmax')) #ouput 


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
ealystopping = EarlyStopping(monitor='loss', patience=30, mode='auto')


strart_time = datetime.datetime.now()
model.fit(x_train, y_train, epochs=100, callbacks=[ealystopping], verbose=1, validation_split=0.2, batch_size=4)
print('학습 종료시간: ', datetime.datetime.now() - strart_time)
print(x_train.shape, x_test.shape)#
print(y_train.shape, y_test.shape)#
# #평가
loss, acc= model.evaluate(x_test, y_test, batch_size=4)

print("loss : ", loss)
print("acc : ",acc)



# # model.summary()