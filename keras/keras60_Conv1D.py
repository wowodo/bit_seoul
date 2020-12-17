import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
#1 데이터 

a = np.array(range(1, 101))
size = 5

#split_x 멋진 함수를 데려오고
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) #subset
    print(type(aaa))
    return np.array(aaa)

datasets = split_x(a, size)
print("=================")
# print(datasets)

x = datasets[:, 0:4]
y = datasets[:, 4]

print(x)# 이렇게 뽑아보면 1부터 99까지 나오고 97부터 예측값

print(x.shape)# (95, 4) 95 *4 = 380   95,2,2
print(y.shape)# (95,)

#.데이터 전처리
#x범위를 축소시키는 식
from sklearn.preprocessing import MinMaxScaler #
scaler = MinMaxScaler() #0~1 사이에 넣는다.
scaler.fit(x)
#fit 에서 x의 최대값을 뽑고 아래 트렌스폼에서 그 최대 값으로 나눠버린다
x = scaler.transform(x)

x = x.reshape(x.shape[0], 2, 2)
y = y.reshape(y.shape[0], 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True
)

model = Sequential()
model.add(Conv1D(30,2, padding='same', input_shape=(2, 2))) # *****
model.add(Conv1D(50,2, padding='same')) #default activation = linear
model.add(Conv1D(70,2, padding='same',activation='relu'))
model.add(MaxPooling1D())                          # 3, 3, 7 맥스 풀링 통과하면 반으로 \
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1)) #output: 1개


model.summary()

#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.3,
    epochs=100, batch_size=10
)



#4. 평가, 예측

#101.0

loss, mse = model.evaluate(x_test, y_test)
print("loss : ",loss)
print("mse : ", mse)


x_predict = np.array([97, 98, 99, 100])
x_predict = scaler.transform([x_predict])
x_predict = x_predict.reshape(1, 2, 2)

y_predict = model.predict(x_predict)
print("예측값: ", y_predict)

'''
loss :  9.152208804152906e-05
mse :  9.152208804152906e-05
예측값:  [[101.00249]]
'''

