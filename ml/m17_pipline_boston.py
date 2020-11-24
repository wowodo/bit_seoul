import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes,load_boston,load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline



#1.데이터

dataset = load_boston()

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.7
)

parameters = [
    {"malddong__n_estimators"     : [100, 200],  # 총 경우에 수가 다 곱하기 좋은것으로 
    "malddong__max_depth"         : [6, 8, 10, 12],
    "malddong__min_samples_leaf"  : [3, 5, 7, 10],
    "malddong__min_samples_split" : [2, 3, 5, 10],
    "malddong__n_jobs"            : [-1]}
]


#1.데이터

#.2모델
pipe = Pipeline([("scaler", MinMaxScaler()), ('malddong',RandomForestClassifier())])  #svc 모델을 쓰는데 민맥스 스케일러를 쓰겠다\ 

model = RandomizedSearchCV(pipe, parameters, verbose=2,  cv=5)

#3. 훈련
model.fit(x_train, y_train)

print('acc : ', model.score(x_test, y_test)) 

#평가예측
print("최적의 매개변수 : ", model.best_estimator_)

y_predict = model.predict(x_test)
print("최종 정답률 : ", r2_score(y_test, y_predict))
