from sklearn.datasets import load_wine
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



#1.데이터
x, y = load_wine(return_X_y = True)

# print(datasets.feature_names)
# print(datasets.target_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

scale= StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# model = KNeighborsClassifier()
# # model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()


# #3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test) 
# # accuracy_score 를 넣어서 비교할것 #분류 모델 일 때는
# # 회기 모델일 경우  r2_score 와 비교할것
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

# # r2 = r2_score(y_test, y_predict)
# # print("r2_score : ", r2)

print(y_test[ :10], '의 예측 결과', '\n', y_predict[ :10])


# '''
# accuracy로 

# model = RandomForestClassifier()
# --------------------------------
# accuracy_score :  1.0
# [1 1 0 0 1 0 1 1 0 0] 의 예측 결과
#  [1 1 0 0 1 0 1 1 0 0]
# -------------------------------

# model = KNeighborsClassifier()
# --------------------------------
# accuracy_score :  0.9444444444444444
# [1 1 0 0 1 0 1 1 0 0] 의 예측 결과
#  [1 1 0 0 1 0 1 2 0 0]
# --------------------------------

# SVC
# --------------------------------
# accuracy_score :  1.0
# [1 1 0 0 1 0 1 1 0 0] 의 예측 결과
#  [1 1 0 0 1 0 1 1 0 0]
# --------------------------------

# model = LinearSVC()
# --------------------------------
# accuracy_score :  0.9722222222222222
# [1 1 0 0 1 0 1 1 0 0] 의 예측 결과
#  [1 1 0 0 1 0 1 1 0 0]
# --------------------------------



# r2로 

# model = KNeighborsRegressor()
# -------------------------------
# 에러
# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
# --------------------------

# model = RandomForestRegressor()
# ------------------------------
# 에러
# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
# ------------------------------


# score  어쩌고
# '''