'''
#sklearn 2.3.1 에서 에러
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/csv/boston_house_prices.csv',header=1, index_col=0) 

x = iris.iloc[:, :-1]
y = iris.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

allAlgorithms = all_estimators(type_filter='regressor')#클래스 파이어 모델들을

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))


    except:
        pass
        
        
import sklearn
print(sklearn.__version__)#0.22.1버전에 문제 있어서 출력이 안됨 -> 버전 낮춰 이용

#
#D:\Study> pip uninstall scikit-learn 0.23.2
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=0)

x = iris.iloc[:, :-1]
y = iris.iloc[:, -1:]




x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)

allAlgorithms = all_estimators(type_filter='regressor') #이걸 지원을 안 함


for (name, algorithm) in allAlgorithms: 
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 
                            #try, catch 사용해서 소스 완성하기 (에러 건너뛰기)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', r2_score(y_test, y_pred))

    except:
        pass

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함