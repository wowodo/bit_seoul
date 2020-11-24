from sklearn.svm import LinearSVC,  SVC
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#accuracy

#1.데이터
x_data = [[0, 0], [1, 0], [0, 1],[1, 1]]
y_data =[0, 1 ,1 ,0]

#모델
# model = LinearSVC()
# model =SVC()

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))


#3훈련
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4평가

y_predict = model.predict(x_data)
print(x_data, "의 예측결과", y_predict) #[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [0 0 0 1]

acc1 = model.evaluate(x_data, y_data)
print("model.evaluate", acc1) #스코어는 데이터로

# acc2 = accuracy_score(y_data, y_predict)
# print("accuracy_score : ",acc2) # 에러남 해결해
