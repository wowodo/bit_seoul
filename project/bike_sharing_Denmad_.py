
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV

bike_x = np.load('./project/bike_x.npy')
bike_y = np.load('./project/bike_y.npy')


parameters = [
    {"bike__n_estimators": [100, 500, 200], "bike__learning_rate": [10, 21, 13, 44], "bike__max_depth":[4, 5, 6]},
    {"bike__n_estimators": [90, 200, 300], "bike__learning_rate": [5, 10, 10], "bike__max_depth":[4, 5, 6], "bike__colsample_bytree": [0.6, 0.9, 1]},
    {"bike__n_estimators": [90, 110], "bike__learning_rate": [0.1, 0.001, 0.5], "bike__max_depth":[4, 5, 6], "bike__colsample_bytree":[0.6, 0.9, 1],
    "bike__colsample": [0.9, 0.8, 0.9]}
]

# PCA

# pca = PCA()
# pca.fit(bike_x)

#  = np.cunsum(pca.explained_variance_ratio_)
# print(cumsum)

# d = np.cumsum(cumsum >= 0.95) +1
# pca = PCA(n_components=d)
# print(cumsum)



x_train, x_test, y_train, y_test = train_test_split(bike_x,bike_y, train_size=0.7, random_state=66,shuffle=True)

print(x_train.shape)

model = XGBRegressor(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9766081871345029

print(model.feature_importances_)


ii = []
for i in range(bike_x.data.shape[1]):
    if model.feature_importances_[i] == 0:
        print(i,"번째 컬럼은 0")
        ii.append(i)
    else:
        print(i," 번째 컬럼은 0아님")

print(ii) # [0, 2, 5, 25]

for i in range(len(ii)):
    x_train = np.delete(x_train,ii[i],axis=1)
    x_test  = np.delete(x_test,ii[i],axis=1)
    print(x_train.shape)
    print(x_test.shape)


pca = PCA()
pca.fit(x_train)
# 중요도가 높은순서대로 바뀐다.
pca.explained_variance_ratio_
xd = len(pca.explained_variance_ratio_)
print(xd)
xd = int((xd*0.7))
print(xd)
pca = PCA(n_components=xd)

x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
model = XGBRegressor(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
#acc :  0.007518796992481203

# # model = RandomizedSearchCV(XGBRegressor(),parameters, cv=5, verbose=2)
# model = GridSearchCV(XGBRegressor(),parameters, cv=5, verbose=2)

# model.fit(x_train, y_train)
# print(' 최적의 매개변수 :', model.best_estimator_)
# acc = model.score(x_test,y_test)

# print("acc : ",acc)



'''
pipeline
PCA
피쳐임폴턴트
셀렉트 모델.

:  0.2754622891793961
'''
'''
그리드
acc :  0.24654251789156134
랜덤
acc :  0.2462712882123509
'''