# winequality-white.csv

#RF로 모델을 만들것

from sklearn.datasets import load_wine
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust = 이상치 제거에 효과적
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



#1.데이터
data_np = np.loadtxt('./data/csv/winequality-white.csv', delimiter=';', skiprows=1) #head 제외하고 읽음


# print(type(data_np))


#불러온 데이터를 판다스로 저장(*.csv)하시오.
# data_pd = pd.DataFrame(data_np)
# data_pd.to_csv('./data/csv/winequality-white.csv')


# print(datasets.feature_names)
# print(datasets.target_names)

# print(data_np.shape) #(150, 5)
# print(type(data_np))

x = data_np[:,: data_np.shape[1]-1]
y = data_np[:, data_np.shape[1]-1:]


#
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=56, shuffle=True, train_size=0.8
)

scale= StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# #2.모델 테스트 할때 하나씩 해보기
# # model = LinearSVC()
# # model = SVC()
# model = KNeighborsClassifier()
# # model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()


# #3. 훈련
model.fit(x_train, y_train)

score = model.score(x_test, y_test) 
print("model.score: ", score)
# # accuracy_score 를 넣어서 비교할것 #분류 모델 일 때는
# # 회기 모델일 경우  r2_score 와 비교할것
y_predict = model.predict(x_test)
metrics_score = accuracy_score(y_test, y_predict)
print("metrics_score : ",metrics_score)



# # r2 = r2_score(y_test, y_predict)
# # print("r2_score : ", r2)

print(y_test[ :10], '의 예측 결과', '\n', y_predict[ :10])

'''
model.score:  0.6775510204081633
metrics_score :  0.6775510204081633
'''


