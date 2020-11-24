#실습
#유방암
#당뇨병
#보스톤  RandomForestRegressor()
#와인
#파일을   randomSearch2, 3, 4, 5,
#RandomForest로 만들기

#유방암 데이터
#모델 : RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# parameters = [
#     {"n_estimators"       : [100, 200]},# 둘중에 좋은 걸로 선택 
#     {"max_depth"         : [6, 8, 10, 12]},
#     {"min_samples_leaf"  : [3, 5, 7, 10]},
#     {"min_samples_split" : [2, 3, 5, 10]},
#     {"n_jobs"            : [-1]}
# ]


parameters = [
    {"n_estimators"     : [100, 200],  # 총 경우에 수가 다 곱하기 좋은것으로 
    "max_depth"         : [6, 8, 10, 12],
    "min_samples_leaf"  : [3, 5, 7, 10],
    "min_samples_split" : [2, 3, 5, 10],
    "n_jobs"            : [-1]}
]


#1.데이터

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.7
)

# scale= StandardScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

#2.모델 테스트 할때 하나씩 해보기
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

kfold = KFold(n_splits=5, shuffle=True)
model = SVC()
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)

#3. 훈련
model.fit(x_train, y_train)

#평가예측
print("최적의 매개변수 : ",model.best_estimator_)


y_predict = model.predict(x_test)
print("최종 정답률     : ",accuracy_score(y_test,y_predict))


'''
# parameters = [
#     {"n_estimators"       : [100, 200]},# 둘중에 좋은 걸로 선택 
#     {"max_depth"         : [6, 8, 10, 12]},
#     {"min_samples_leaf"  : [3, 5, 7, 10]},
#     {"min_samples_split" : [2, 3, 5, 10]},
#     {"n_jobs"            : [-1]}
# ]

최적의 매개변수 :  RandomForestClassifier(n_estimators=200)
최종 정답률     :  0.9707602339181286
'''

'''

parameters = [
    {"n_estimators"     : [100, 200],  # 총 경우에 수가 다 곱하기 좋은것으로 
    "max_depth"         : [6, 8, 10, 12],
    "min_samples_leaf"  : [3, 5, 7, 10],
    "min_samples_split" : [2, 3, 5, 10],
    "n_jobs"            : [-1]}
]

최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=5,
                       n_jobs=-1)
최종 정답률     :  0.9649122807017544

#########################RandomSearch#########################

최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200,
                       n_jobs=-1)
최종 정답률     :  0.9707602339181286
'''