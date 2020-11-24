import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score #모델에 관여 KFold, cross_val_score 


# 1.데이터
iris= pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[: , :4] #(150, 4) 
y = iris.iloc[: , -1] #(150,)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, r2_score  #분류  acc 회귀 r2

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



#1.데이터
x, y = load_iris(return_X_y = True)


# print(datasets.feature_names)
# print(datasets.target_names)

# x = dataset.data
# y = dataset.taget

# scale= StandardScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

#2.모델

kfold = KFold(n_splits=5, shuffle=True)       #5조각을 내고 그것을 섞겠다

model = SVC()
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print('scores :', scores)

# #3. 훈련
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test) 
# # accuracy_score 를 넣어서 비교할것 #분류 모델 일 때는
# # 회기 모델일 경우  r2_score 와 비교할것
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)


# # r2 = r2_score(y_test, y_predict)
# # print("r2_score : ", r2)

# print(y_test[ :10], '의 예측 결과', '\n', y_predict[ :10])
