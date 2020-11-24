#분류

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)


kfold = KFold(n_splits=5, shuffle=True)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=56
)

# # #2.모델 테스트 할때 하나씩 해보기
# # # model = LinearSVC()
# model = SVC()
# # model = KNeighborsClassifier()
# # # model = KNeighborsRegressor()
# # model = RandomForestClassifier()
# # model = RandomForestRegressor()
# scores = cross_val_score(model, x_train, y_train, cv=kfold)


allAlgorithms = all_estimators(type_filter='classifier') #이걸 지원을 안 함

for (name, algorithm) in allAlgorithms: 
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 

        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', accuracy_score(y_test, y_pred))


    except:
        # pass    # contiune
        print(name, "None!")

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함 
print('scores :', scores)


'''
AdaBoostClassifier 의 정답률:  0.9
BaggingClassifier 의 정답률:  0.9333333333333333
BernoulliNB 의 정답률:  0.3333333333333333
CalibratedClassifierCV 의 정답률:  0.9333333333333333
CategoricalNB 의 정답률:  0.9
CheckingClassifier 의 정답률:  0.3333333333333333
ClassifierChain 은 없는 놈!!
ComplementNB 의 정답률:  0.6666666666666666
DecisionTreeClassifier 의 정답률:  0.9333333333333333
DummyClassifier 의 정답률:  0.26666666666666666
ExtraTreeClassifier 의 정답률:  0.9333333333333333
ExtraTreesClassifier 의 정답률:  0.9333333333333333
GaussianNB 의 정답률:  0.9666666666666667
GaussianProcessClassifier 의 정답률:  0.9333333333333333
GradientBoostingClassifier 의 정답률:  0.9333333333333333
HistGradientBoostingClassifier 의 정답률:  0.9333333333333333
KNeighborsClassifier 의 정답률:  0.9666666666666667
LabelPropagation 의 정답률:  0.9666666666666667
LabelSpreading 의 정답률:  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률:  0.9666666666666667
LinearSVC 의 정답률:  0.9333333333333333
LogisticRegression 의 정답률:  0.9333333333333333
LogisticRegressionCV 의 정답률:  0.9333333333333333
MLPClassifier 의 정답률:  0.9666666666666667
MultiOutputClassifier 은 없는 놈!!
MultinomialNB 의 정답률:  0.9333333333333333
NearestCentroid 의 정답률:  0.9333333333333333
NuSVC 의 정답률:  0.9333333333333333
OneVsOneClassifier 은 없는 놈!!
OneVsRestClassifier 은 없는 놈!!
OutputCodeClassifier 은 없는 놈!!
PassiveAggressiveClassifier 의 정답률:  0.9
Perceptron 의 정답률:  0.8333333333333334
QuadraticDiscriminantAnalysis 의 정답률:  0.9333333333333333
RadiusNeighborsClassifier 의 정답률:  0.9666666666666667
RandomForestClassifier 의 정답률:  0.9333333333333333
RidgeClassifier 의 정답률:  0.9333333333333333
RidgeClassifierCV 의 정답률:  0.9333333333333333
SGDClassifier 의 정답률:  0.6666666666666666
SVC 의 정답률:  0.9333333333333333
StackingClassifier 은 없는 놈!!
VotingClassifier 은 없는 놈!!
0.23.1
scores : [0.95833333 1.         0.91666667 0.91666667 0.95833333]
'''