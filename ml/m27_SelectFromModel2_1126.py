 #실습
# 1. 상단 모델에 그리드서치 또는 랜덤서치 적용
# 최적의 R2값과 피쳐임포턴스 구할것

# 2. 위 쓰레드값으로 SelectFromModel을 구해서 
# 최적의 피쳐갯수를 구할것

# 3. 위 피쳐 갯수로 데이터(피쳐)를 수정(삭제)해서
# 그래드서치 또는 랜덤서치 적용
# 최적의 R2값을 구할 것

# 1번값과 2번값을 비교해볼것

import numpy as np
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV


boston = load_boston()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=0.7, random_state=66,shuffle=True)


parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01], "max_depth":[4,5,6]},
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.001], "max_depth":[4,5,6],
      "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

model = GridSearchCV(XGBRegressor(), parameters, cv=5,verbose=2)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc1 : ",acc)

print("최적의 estimator : ",model.best_estimator_)
print("최적의 params    : ",model.best_params_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",r2_score(y_test,y_predict))


model = model.best_estimator_

thresholds = np.sort(model.feature_importances_)
print(thresholds)

n  = 0
r2 = 0
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 첫번째 돌릴때 첫번째 값 이상을 다 돌려라  두번 돌때는 두번째 값 이상만 돌려라. #thresh 다른 옵션 찾기

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    if score*100.0 > r2:
        n = select_x_train.shape[1]
        r2 = score*100.0
        L_selection = selection
        print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100.0))
    


x_train = L_selection.transform(x_train)
x_test = L_selection.transform(x_test)

model = GridSearchCV(XGBRegressor(), parameters, cv=5,verbose=2)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc1 : ",acc)

print("최적의 estimator : ",model.best_estimator_)
print("최적의 params    : ",model.best_params_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",r2_score(y_test,y_predict))

model = model.best_estimator_

thresholds = np.sort(model.feature_importances_)
print(thresholds)

# 1번



# 2번









