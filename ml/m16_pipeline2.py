#분류
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
from sklearn.svm import LinearSVC, SVC
import warnings
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import  MaxAbsScaler, RobustScaler

#1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=66
)
#svc__ 해줘야 파라미터를 읽는다 
parameters = [
    {"malddong__C":[1, 10, 100, 1000], "malddong__kernel":["linear"]},
     {"malddong__C":[1, 10, 100, 1000], "malddong__kernel":["rbf"]    , "malddong__gamma":[0.001, 0.0001]},
     {"malddong__C":[1, 10, 100, 1000], "malddong__kernel":["sigmoid"], "malddong__gamma":[0.001, 0.0001]}
]

#크로스 발리데이션에서 과적합을 피하기 위해 스케일을 엮는다

#.2모델
pipe = Pipeline([("scaler", MinMaxScaler()), ('malddong',SVC())])  #svc 모델을 쓰는데 민맥스 스케일러를 쓰겠다\ 

model = RandomizedSearchCV(pipe, parameters, cv=5) #모델에 들어온건 cv 크로스 발리데이션 적용한다 
#3.훈련
model.fit(x_train, y_train)# 크로스 발리데이션cv 을 적용해서 훈련한다 트렌스 폼은 자동으로 한다

print('acc : ', model.score(x_test, y_test)) #acc :  1.0

print(' 최적의 매개변수 :', model.best_estimator_)

'''
acc :  1.0
 최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('svc', SVC(C=1000, gamma=0.001))])


                
'''

#파이프 라인 스케일링 엮기 크로스 발리데이션에서는 스케일링 을 안해줬다
#



