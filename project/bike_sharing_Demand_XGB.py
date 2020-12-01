
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, mean_squared_log_error

#블러오기
bike_x = np.load('./project/bike_x.npy')
bike_y = np.load('./project/bike_y.npy')
bike_test = np.load('./project/bike_test_x.npy',allow_pickle=True)
bike_sample = np.load('./project/bike_sample.npy',allow_pickle=True)

parameters = [
    {"bike__n_estimators": [100, 500, 200], "bike__learning_rate": [0.2, 0.5, 0.5, 0.2], "bike__max_depth":[4, 5, 6]},
    {"bike__n_estimators": [90, 200, 300], "bike__learning_rate": [0.3, 0.56, 0.5], "bike__max_depth":[4, 5, 6], "bike__colsample_bytree": [0.6, 0.9, 1]},
    {"bike__n_estimators": [90, 110], "bike__learning_rate": [0.1, 0.001, 0.5], "bike__max_depth":[4, 5, 6], "bike__colsample_bytree":[0.6, 0.9, 1],
    "bike__colsample": [0.9, 0.8, 0.9]}
]

n_jobs = -1 #cpu로 모든 코어를 다쓴다
# 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있습니다
x_train, x_test, y_train, y_test = train_test_split(bike_x,bike_y, train_size=0.7, random_state=66,shuffle=True)

#파이프 라인 
pipe = Pipeline([("scaler", StandardScaler()), ('bike',XGBRegressor())])#스케일러를 추가 할수 있다 
# StandardScaler 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
#이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

kfold = KFold(n_splits=5, shuffle=True)

# model = RandomizedSearchCV(XGBRegressor(),parameters, cv=5, verbose=2)
model = RandomizedSearchCV(XGBRegressor(),parameters, verbose=2)

model.fit(x_train, y_train)
print(' 최적의 매개변수 :', model.best_estimator_)
acc = model.score(x_test,y_test)

print("acc : ",acc)

y_predict = model.predict(bike_test)
print(y_predict[-10:])
print(y_predict.shape)

model = model.best_estimator_ #최적의 XGBRegressor 

thresholds = np.sort(model.feature_importances_) #모델의 컬럼의 중요도를 뽑는다 np.sort (가장 낮은 순으로) 모델 핏까지 한번 돌려야지 피쳐임폴턴트를 뽑을 수 있다.,
print(thresholds)

n     = 0
b_acc = 0

#selectfromModel

for thresh in thresholds: #thresholds에서 하나씩 뽑아서 thresh 널어 준다
    selection = SelectFromModel(model, threshold=thresh, prefit=True)#SelectFromModel에 모델과 컬럼 중요도를 넣어주면
    #낮은 순으로 들어가 있는 것을 model 에서 하나씩 빼서 셀렉션으로 넣어 준다

    select_x_train = selection.transform(x_train) #x_train은 자르고 남은 수가 된다. 그것을 slect_x_train으로 만들어 준다
    selection_model = XGBRegressor()#새로운 모델을 생성한다
    selection_model.fit(select_x_train,y_train)#남은 컬럼으로 모델 핏을 해본다

    select_x_test = selection.transform(x_test)#위에 x_train 과 같이 자르고 남을것을 select_x_test에 넣어준다
    y_predict = selection_model.predict(select_x_test)# select_x_test 으로 바꾼 값을 예측 해본다.

    acc = selection_model.score(select_x_test,y_test) #select_x_test와 y_test의 스코어를 매긴다
    if acc > b_acc:
        n = select_x_train.shape[1]
        b_acc = acc
        L_selection = selection
        print("Thresh=%.3f, n=%d, acc: %.15f%%"%(thresh,select_x_train.shape[1],acc))

x_train = L_selection.transform(x_train)
x_test = L_selection.transform(x_test)

model = RandomizedSearchCV(pipe,parameters, cv=5, verbose=2)

model.fit(x_train, y_train)
print(' 최적의 매개변수 :', model.best_estimator_)
acc = model.score(x_test,y_test)

model = model.best_estimator_# RandomizedSearchCV 돌면 꼭 model = model.best_estimator_해서 값을 뽑아 줘야 한다

bike_test = L_selection.transform(bike_test)

y_predict = model.predict(bike_test)
print(y_predict[-10:])
print(y_predict.shape)
print("acc : ",acc)

tmp_predict = model.predict(x_test)

y_test =y_test.reshape(y_test.shape[0],)
tmp_predict = tmp_predict.astype('int64')
print(type(y_test[0]))
print(tmp_predict)

print(y_test.shape)
print(tmp_predict.shape)

for i in tmp_predict:
    if i < 0:
        tmp_predict[i] = 0

for i in y_test:
    if i < 0:
        tmp_predict[i] = 0

print(np.sqrt(mean_squared_log_error(y_test, tmp_predict)))

'''
pipeline
PCA
피쳐임폴턴트
셀렉트 모델.
'''
'''
XGB
acc :  0.29161457452861483

selectFromMolde하고 나서 
Thresh=0.068, n=6, acc: 0.300796197269297%

pipe
acc :  0.344353694008916

64.36742  54.078117 61.2505   54.078117 64.36742  61.2505   61.2505
 49.14192  52.25884  52.25884 
'''

#엑스지 부스터를 파이프 라인으로 덮어 우고 그 파이프 라인을 랜덤 서치를 돌린다