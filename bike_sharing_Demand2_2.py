import numpy as np
import pandas as pd

x = np.load('./project/bike_x.npy', allow_pickle=True)
y = np.load('./project/bike_y.npy', allow_pickle=True)

# print(x_train.shape)#(10886, 6)
# print(y_train .shape)#(10886, 1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True 
)


#스케일러는 무족건 2차원만 받기때문에  reshpe  해주려면 알아서
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

size= 5

def new_split (r_data,r_size):
    new_data = []
    for i in range(len(r_data)-(r_size-1)) : 
        new_data.append(r_data[i:i+size])

    return np.array(new_data)

x_train = new_split(x_train,size)
x_test = new_split(x_test,size)

print(x_train)
print(x_train.shape)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)


y_train = y_train[4:]
y_test = y_test[4:]

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input,GRU

model = Sequential()
model.add(LSTM(20, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(400,activation='relu'))
model.add(Dense(250,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))


# print(x_train)
# print(y_test
#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=100, batch_size=50
)



#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test, batch_size=1)

x_predict = x_test[:100]
y_test = y_test[:100]

print(x_predict.shape)
y_predict = model.predict(x_predict)
print(y_predict.shape)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
print("mse: ", mse)
print("mae: ", mae)
