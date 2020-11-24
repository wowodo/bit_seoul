from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#accuracy

#1.데이터
x_data = [[0, 0], [1, 0], [0, 1],[1, 1]]
y_data =[0, 0 ,0 ,1]

#모델
model = LinearSVC()

#3훈련

model.fit(x_data, y_data)

#4평가

y_predict = model.predict(x_data)
print(x_data, "의 예측결과", y_predict) #[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [0 0 0 1]

acc1 = model.score(x_data, y_predict)
print("model.score", acc1) #에러 메시지 뜨니까 처리

acc2 = accuracy_score(y_data, y_predict)
print("accuracy_score : ",acc2) # accuracy_score :  1.0

