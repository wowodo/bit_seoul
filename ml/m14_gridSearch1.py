#분류

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
from sklearn.svm import LinearSVC, SVC
import warnings
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')


iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=66
)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1, 10, 100, 1000], "kernel":["rbf"]    , "gamma":[0.001, 0.0001]},
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

#20번 훈련하면서 핏할때 가장 적합한것도 확인해준다


# 2. 모델
kfold = KFold(n_splits=5, shuffle=True)
# model = SVC()
#그리드 서치 하이퍼 파라미터를 싹 모아서 (전체를 한번에 fit 시키겠다)  GridSearchCV
model = GridSearchCV(SVC(),parameters, cv=kfold)# SVC라는 모델을 그리드로 파라미터 크로스 발리데이션으로 전부 확인 )

# 3. 훈련 
model.fit(x_train, y_train)

#4. 평가 예측
print("최적의 매개변수",model.best_estimator_)#estimator결과자

y_predict = model.predict(x_test)
print(" 최종정담률 : ", accuracy_score(y_test, y_predict))

'''
최적의 매개변수 SVC(C=1, kernel='linear')
 최종정담률 :  0.9666666666666667

'''