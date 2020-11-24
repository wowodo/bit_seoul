#데이터

import numpy as np
import pandas as pd

#1_1. samsung
samsung_x = np.load('./samsung/samsung_x.npy', allow_pickle=True).astype('float32')
samsung_y = np.load('./samsung/samsung_y.npy', allow_pickle=True).astype('float32')
bit_x = np.load('./samsung/bit_x.npy', allow_pickle=True).astype('float32')
bit_y = np.load('./samsung/bit_y.npy', allow_pickle=True).astype('float32')
gold_x = np.load('./samsung/gold_x.npy', allow_pickle=True).astype('float32')
gold_y = np.load('./samsung/gold_y.npy', allow_pickle=True).astype('float32')
kos_x = np.load('./samsung/kos_x.npy', allow_pickle=True).astype('float32')
kos_y = np.load('./samsung/kos_y.npy', allow_pickle=True).astype('float32')



'''
(615, 5, 5)
(615, 1)
(1195, 5, 3)
(1195, 1)
(805, 5, 4)
(805, 1)
(875, 5, 6)
(875, 1)
'''

size = 5

samsung_x = samsung_x[:samsung_x.shape[0], :, :size-1]
samsung_y = samsung_y[:samsung_x.shape[0]]

# print(samsung_x.shape)
# print(samsung_y.shape)




samsung_x_predict = samsung_x[-1:, :, :]


samsung_x = samsung_x[:-2, :, :]
samsung_y = samsung_y[:-2, :]


gold_x = gold_x[:samsung_x.shape[0], :, :size-1]


kos_x = kos_x[:samsung_x.shape[0], :, :size-1]

bit_x = bit_x[:samsung_x.shape[0], :, :size-1]




bit_x_predict = bit_x[-1:, :, :]
gold_x_predict = gold_x[-1:, :, :]
kos_x_predict = kos_x[-1:, :, :]



print(samsung_x.shape)
print(samsung_y.shape)

print(bit_x.shape)
print(bit_y.shape)

print(gold_x.shape)
print(gold_y.shape)

print(kos_x.shape)
print(kos_y.shape)


#train / test
from sklearn.model_selection import train_test_split

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(
    samsung_x, samsung_y, train_size=0.7
)

bit_x_train, bit_x_test = train_test_split(
    bit_x, train_size=0.7
)


gold_x_train, gold_x_test = train_test_split(
    gold_x, train_size=0.7
)

kos_x_train, kos_x_test = train_test_split(
    kos_x, train_size=0.7
)


print(samsung_x.shape)
print(samsung_y.shape)

print(bit_x.shape)


print(gold_x.shape)


print(kos_x_train.shape)



#2. 모델
#import 빠뜨린 거 없이 할 것
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout

#모델 1_samsung
input1 = Input(shape=(samsung_x_train.shape[1], samsung_x_train.shape[2]))
dense1_1 = LSTM(100, activation='relu')(input1)
dense1_2 = Dense(500, activation='relu')(dense1_1)
dense1_3 = Dense(600, activation='relu')(dense1_2)
dense1_4 = Dense(200, activation='relu')(dense1_3)
dense1_5 = Dense(30, activation='relu')(dense1_4)
output1 = Dense(1)(dense1_5)
model1 = Model(inputs=input1, outputs=output1)



#모델2_bit
input2 = Input(shape=(bit_x_train.shape[1], bit_x_train.shape[2]))
dense2_1 = LSTM(40, activation='relu')(input2)
dense2_3 = Dense(256, activation='relu')(dense2_1)
dense2_4 = Dense(1024, activation='relu')(dense2_3)
dense2_5 = Dense(200, activation='relu')(dense2_4)
dense2_6 = Dense(32, activation='relu')(dense2_5)
dense2_7 = Dense(10, activation='relu')(dense2_6)
output2 = Dense(1)(dense2_7)
model2 = Model(inputs=input2, outputs=output2)


input3 = Input(shape=(gold_x_train.shape[1], gold_x_train.shape[2]))
dense3_1 = LSTM(80, activation='relu')(input3)
dense3_3 = Dense(255, activation='relu')(dense3_1)
dense3_4 = Dense(2024, activation='relu')(dense3_3)
dense3_5 = Dense(200, activation='relu')(dense3_4)
dense3_6 = Dense(32, activation='relu')(dense3_5)
dense3_7 = Dense(10, activation='relu')(dense3_6)
output3 = Dense(1)(dense3_7)
model3 = Model(inputs=input3, outputs=output3)


input4 = Input(shape=(kos_x_train.shape[1], kos_x_train.shape[2]))
dense4_1 = LSTM(80, activation='relu')(input4)
dense4_3 = Dense(255, activation='relu')(dense4_1)
dense4_4 = Dense(2024, activation='relu')(dense4_3)
dense4_5 = Dense(200, activation='relu')(dense4_4)
dense4_6 = Dense(32, activation='relu')(dense4_5)
dense4_7 = Dense(10, activation='relu')(dense4_6)
output4 = Dense(1)(dense4_7)
model4 = Model(inputs=input3, outputs=output3)

#병합 
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2,output3,output4])
middle1 = Dense(300, activation='relu')(merge1)
middle1 = Dense(2000, activation='relu')(middle1)
output1 = Dense(800, activation='relu')(output1)
output1 = Dense(30, activation='relu')(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1, input2, input3,input4], outputs=output1)

#3. 컴파일, 훈련

#early stopping
modelpath='./model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

cp = ModelCheckpoint(filepath=modelpath, 
                     monitor='val_loss', 
                     save_best_only=True, 
                     mode='auto'
)


model.fit(
    [samsung_x_train,bit_x_train, gold_x_train,kos_x_train],
    samsung_y_train,
    callbacks=[early_stopping,cp],
    validation_split=0.2,
    epochs=100, batch_size=32
)

model.save_weights('./save/samsung.h5')

#4. 평가

result = model.evaluate(
    [samsung_x_test,bit_x_test, gold_x_test,kos_x_test],
    samsung_y_test,
    batch_size=32)


x_predict = model.predict([samsung_x_predict, bit_x_predict, gold_x_predict, kos_x_predict])
print("loss: ", result[0])
print("예측값: ", x_predict)
