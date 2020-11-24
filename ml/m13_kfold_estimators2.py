#회귀


import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=0)

kfold = KFold(n_splits=5, shuffle=True)

x = iris.iloc[:, :-1]
y = iris.iloc[:, -1:]




x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=
)



allAlgorithms = all_estimators(type_filter='regressor') #이걸 지원을 안 함classifier


for (name, algorithm) in allAlgorithms: 
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 
                            #try, catch 사용해서 소스 완성하기 (에러 건너뛰기)

        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', r2_score(y_test, y_pred))

    except:
        # pass    # contiune
        print(name, "은 없는 놈!!")

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함

print('scores :', scores)

'''
ARDRegression 의 정답률:  0.7413660842741397
AdaBoostRegressor 의 정답률:  0.8382440303997568
BaggingRegressor 의 정답률:  0.857671136060465
BayesianRidge 의 정답률:  0.7397243134288036
CCA 의 정답률:  0.7145358120880194
DecisionTreeRegressor 의 정답률:  0.8474609193556001
DummyRegressor 의 정답률:  -0.0007982049217318821
ElasticNet 의 정답률:  0.6952835513419808
ElasticNetCV 의 정답률:  0.6863712064842076
ExtraTreeRegressor 의 정답률:  0.8319694216013195
ExtraTreesRegressor 의 정답률:  0.8944108147312025
GammaRegressor 은 없는 놈!!
GaussianProcessRegressor 의 정답률:  -5.586473869478007
GeneralizedLinearRegressor 은 없는 놈!!
GradientBoostingRegressor 의 정답률:  0.8992659091891542
HistGradientBoostingRegressor 의 정답률:  0.8843141840898427
HuberRegressor 의 정답률:  0.7650865977198575
IsotonicRegression 은 없는 놈!!
KNeighborsRegressor 의 정답률:  0.6550811467209019
KernelRidge 의 정답률:  0.7635967086119403
Lars 의 정답률:  0.7440140846099281
LarsCV 의 정답률:  0.7499770153318335
Lasso 의 정답률:  0.683233856987759
LassoCV 의 정답률:  0.7121285098074346
LassoLars 의 정답률:  -0.0007982049217318821
LassoLarsCV 의 정답률:  0.7477692079348518
LassoLarsIC 의 정답률:  0.74479154708417
LinearRegression 의 정답률:  0.7444253077310314
LinearSVR 의 정답률:  0.696335766181126
MLPRegressor 의 정답률:  0.5370608044272749
MultiOutputRegressor 은 없는 놈!!
MultiTaskElasticNet 의 정답률:  0.6952835513419808
MultiTaskElasticNetCV 의 정답률:  0.6863712064842077
MultiTaskLasso 의 정답률:  0.6832338569877592
MultiTaskLassoCV 의 정답률:  0.7121285098074348
NuSVR 의 정답률:  0.32492104048309933
OrthogonalMatchingPursuit 의 정답률:  0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률:  0.7377665753906504
PLSCanonical 의 정답률:  -1.3005198325202088
PLSRegression 의 정답률:  0.7600229995900802
PassiveAggressiveRegressor 의 정답률:  -0.2153983910601549
PoissonRegressor 은 없는 놈!!
RANSACRegressor 의 정답률:  0.7276260084372401
RadiusNeighborsRegressor 은 없는 놈!!
RandomForestRegressor 의 정답률:  0.8903847636545057
RegressorChain 은 없는 놈!!
Ridge 의 정답률:  0.7465337048988421
RidgeCV 의 정답률:  0.7452747021926976
SGDRegressor 의 정답률:  -2.6319218710974963e+26
SVR 의 정답률:  0.2867592174963418
StackingRegressor 은 없는 놈!!
'''