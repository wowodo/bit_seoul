#당뇨병
#보스톤  RandomForestRegressor()
#와인
#파일을   gridSearch3,4,5,
#RandomForest로 만들기
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {"n_estimators"     : [100, 200],  # 총 경우에 수가 다 곱하기 좋은것으로 
    "max_depth"         : [6, 8, 10, 12],
    "min_samples_leaf"  : [3, 5, 7, 10],
    "min_samples_split" : [2, 3, 5, 10],
    "n_jobs"            : [-1]}
]

#1.데이터

dataset = load_diabetes()
x = dataset.data
y = dataset.target

#1.데이터

kfold = KFold(n_splits=5, shuffle=True)
model = SVC()
model = RandomizedSearchCV(RandomForestRegressor(), parameters, verbose=2,  cv=kfold)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.7
)


#3. 훈련
model.fit(x_train, y_train)

#평가예측
print("최적의 매개변수 : ",model.best_estimator_)

y_predict = model.predict(x_test)
print("최종 정답률 : " ,r2_score(y_test, y_predict))

'''
최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=10, n_estimators=200,
                      n_jobs=-1)
최종 정답률 :  0.38082550861554143

RandomSearch
최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=10, min_samples_split=3,
                      n_estimators=200, n_jobs=-1)
최종 정답률 :  0.3841794907340955
'''