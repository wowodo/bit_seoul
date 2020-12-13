# 이벨류에이트 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target
# 1.데이터
x, y  = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    shuffle=True,
                                                    random_state=77
)

#2.모델
model = XGBRegressor(n_estimators=2000, learning_rate=0.01)
# model = XGBRegressor(learning_rate=0.01)

'''
n_estimators
디폴트로 100번돈다

n_estimators=10,
10번 훈련
'''

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric=(["rmse","mae"]),  #메트릭스는 평가라 영향을 주지 않는다 rmse mse 에 루트를 씌운것
                            eval_set=[(x_train, y_train),(x_test, y_test)]  #훈련셋과 테스트 벌보스 둘다 볼수 있다 
)

'''
eval_metric=(["rmse","mae"])

[1199]  validation_0-rmse:0.28893       validation_0-mae:0.20988        validation_1-rmse:2.52192       validation_1-mae:1.87063
r2 :  0.911030232620356
'''


# eval_metric의 대표 파람 =  rmse, mae, logloss, error, auc

#
results = model.evals_result()
# print("eval's results : ", results)

#4. 평가,예측
y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)

print("r2 : ", r2)

'''
[1199]  validation_0-rmse:0.28893       validation_1-rmse:2.52192
r2 :  0.911030232620356
'''