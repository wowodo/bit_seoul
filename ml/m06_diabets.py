import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



#1.데이터
x, y = load_diabetes(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

scale= StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

#2.모델 테스트 할때 하나씩 해보기
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test) 
# accuracy_score 를 넣어서 비교할것 #분류 모델 일 때는
# 회기 모델일 경우  r2_score 와 비교할것
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

# r2 = r2_score(y_test, y_predict)
# print("r2_score : ", r2)

print(y_test[ :10], '의 예측 결과', '\n', y_predict[ :10])



'''
accuracy로 

model = RandomForestClassifier()
--------------------------------
accuracy_score :  0.0
[ 48. 131. 101. 277. 108.  64.  66. 152. 281. 215.] 의 예측 결과
 [166.  31.  72. 168. 102.  60. 154. 245. 170. 150.]
-------------------------------

model = KNeighborsClassifier()
--------------------------------
accuracy_score :  0.0
[ 48. 131. 101. 277. 108.  64.  66. 152. 281. 215.] 의 예측 결과
 [113.  31.  70. 124.  66.  40.  50. 109. 151. 109.]
--------------------------------

SVC
--------------------------------
accuracy_score :  0.0
[ 48. 131. 101. 277. 108.  64.  66. 152. 281. 215.] 의 예측 결과
 [202.  90.  71. 178. 102.  90.  90. 109. 178. 109.]
--------------------------------

model = LinearSVC()
--------------------------------
accuracy_score :  0.011235955056179775
[ 48. 131. 101. 277. 108.  64.  66. 152. 281. 215.] 의 예측 결과
 [258.  89. 101. 199. 102.  40. 190. 273. 221. 258.]
--------------------------------



r2로 

model = KNeighborsRegressor()
-------------------------------
에러
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
--------------------------

model = RandomForestRegressor()
------------------------------
에러
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
------------------------------


score  어쩌고
'''