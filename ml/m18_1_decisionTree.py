#decisionTree를 보지말고 RnadomForest  를 봐라 트리구조가 머신러닝에서 가장 좋다
# #트리 구조 공부

# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split


# canser = load_breast_cancer()
# x_train, x_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, random_state=56,train_size=0.8
# )
# #y값으로 클래스 파이언지 다른건지 판단해야 한다
# model = DecisionTreeClassifier(max_depth=4)

# model.fit(x_train,y_train)

# acc = model.score(x_test, y_test)

# print(acc)
# print(model.feature_importances_)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)

model =  DecisionTreeClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9415204678362573
print(model.feature_importances_)

#시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()



