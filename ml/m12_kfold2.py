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



# #1.데이터
x, y = load_iris(return_X_y = True)


# #2.모델

kfold = KFold(n_splits=5, shuffle=True)       #5조각을 내고 그것을 섞겠다


# # #2.모델 테스트 할때 하나씩 해보기
# # # model = LinearSVC()
# model = SVC()
# # model = KNeighborsClassifier()
# # # model = KNeighborsRegressor()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print('SVC :', scores)


'''
SVC : [0.91666667 1.         0.95833333 1.         0.95833333]
'''

# # #2.모델 테스트 할때 하나씩 해보기
# model = LinearSVC()
# # model = SVC()
# # model = KNeighborsClassifier()
# # # model = KNeighborsRegressor()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print('LinearSVC :', scores)
'''
LinearSVC : [0.91666667 0.79166667 0.75       0.83333333 0.95833333]
'''

# # #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print('KNeighborsRegressor() :', scores)
'''
KNeighborsRegressor() : [0.96933661 0.99056511 1.         0.952      0.95990172]
'''

# # #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# # model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print('KNeighborsClassifier :', scores)
'''
andomForestClassifier : [0.95043478 0.9775     0.98235294 1.         0.97538462]
'''

# # #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# # model = KNeighborsClassifier()
# # model = KNeighborsRegressor()
# model = RandomForestClassifier()
# # model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print(' RandomForestClassifier:', scores)
'''
 RandomForestClassifier: [1.         0.95833333 1.         1.         1.        ]
'''

# # #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# # model = KNeighborsClassifier()
# # model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print(' RandomForestClassifier:', scores)
'''
RandomForestRegressor: [0.96180714 0.95596923 0.99955135 0.997472   0.99325   ]
'''
