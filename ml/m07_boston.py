import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



#1.데이터
x, y = load_boston(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

scale= StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

#2.모델 테스트 할때 하나씩 해보기
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test) 
# accuracy_score 를 넣어서 비교할것 #분류 모델 일 때는
# 회기 모델일 경우  r2_score 와 비교할것
y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print("accuracy_score : ", acc)

# r2 = r2_score(y_test, y_predict)
# print("r2_score : ", r2)

print(y_test[ :10], '의 예측 결과', '\n', y_predict[ :10])

'''
accuracy로 

model = RandomForestClassifier()
--------------------------------
에러
ValueError: Unknown label type: 'continuous'
-------------------------------

model = KNeighborsClassifier()
--------------------------------
에러
ValueError: Unknown label type: 'continuous'
--------------------------------

SVC
--------------------------------\
에러
ValueError: Unknown label type: 'continuous'
--------------------------------

model = LinearSVC()
--------------------------------
에러
ValueError: Unknown label type: 'continuous'
--------------------------------



r2로 

model = KNeighborsRegressor()
-------------------------------
accuracy_score :  0.7043729183628615
[ 5.  21.2 22.  23.3 31.5 21.6  7.2 15.1 26.7 31.6] 의 예측 결과
 [ 7.94 22.52 21.34 23.82 30.72 21.88  9.96 14.   24.58 31.66]
--------------------------

model = RandomForestRegressor()
------------------------------
에러
accuracy_score :  0.8440816717302568
[ 5.  21.2 22.  23.3 31.5 21.6  7.2 15.1 26.7 31.6] 의 예측 결과
 [ 9.577 20.97  26.516 26.604 33.121 21.792  8.381 13.559 29.635 32.085]
------------------------------
'''