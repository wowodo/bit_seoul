from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
# cmd 창에서 > pip install xgboost
#설치


cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)


model = XGBClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ",acc)



#파이썬에서 제공하는 것 xgb말고도 다 저장한다.
#로드해서 정상적인 값이 나오면 가중치 까지 저장 했다는 것
import pickle
pickle.dump(model, open("./save/xgb_save/cencer.pickle.dat", "wb"))
print("저장됫다.")

model2 = pickle.load(open("./save/xgb_save/cencer.pickle.dat", "rb"))
print("불러왔다")

acc2 = model2.score(x_test,y_test)
print("acc : ",acc2)