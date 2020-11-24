import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust = 이상치 제거에 효과적
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0) 

y = wine['quality']
x = wine.drop('quality', axis=1)
#x 에서 퀄리티를 빼내겠다

print(x.shape)      #(4898, 11)
print(y.shape)      #(4898,)

newlist =[]
for i in list(y): # 아이의 리스트 순서대로
    if i < 4:
        newlist +=[0]
    elif i <= 7:
        newlist +=[1]
    else :
        newlist +=[2]

y = newlist



#모델 만든거에 이어라
# x = data_np[:,: data_np.shape[1]-1]
# y = data_np[:, data_np.shape[1]-1:]

print(x.shape)  # (4898, 11)
# print(y.shape) #(4898,)

#
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# scale= StandardScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(Dense(30, input_shape= (11 , )))
model.add(Dense(300))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))
# #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# model = KNeighborsClassifier()
# # model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()


# #3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100)

# score = model.score(x_test, y_test) 
# print("model.score: ", score)
# # accuracy_score 를 넣어서 비교할것 #분류 모델 일 때는
# # 회기 모델일 경우  r2_score 와 비교할것

# metrics_score = accuracy_score(y_test, y_predict)
# print("metrics_score : ",metrics_score)

acc = model.evaluate(x_test, y_test)
print(acc)

# # r2 = r2_score(y_test, y_predict)
# # print("r2_score : ", r2)



# y_predict = model.predict(x_test)
# print(y_test[ :10], '의 예측 결과', '\n', y_predict[ :10])

'''
[0.17284640669822693, 0.9591836929321289]
 
'''