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
model.fit(x_train, y_train, verbose=True, eval_metric=["logloss","rmse"],  #메트릭스는 평가라 영향을 주지 않는다 rmse mse 에 루트를 씌운것
                            eval_set=[(x_train, y_train),(x_test, y_test)], #훈련셋과 테스트 벌보스 둘다 볼수 있다 
                            # early_stopping_rounds=20  # 얼리 스탑핑!!! 20번보다 지표가 높으면 정지 지표는  rmse
)

'''
 early_stopping_rounds=20 

[378]   validation_0-rmse:1.31722       validation_1-rmse:2.41937  378번 째에서 제일 좋았다.

r2 :  0.9115357819172423
'''


# eval_metric의 대표 파람 =  rmse, mae, logloss, error, auc

#
results = model.evals_result()
# print("eval's results : ", results)

#4. 평가,예측


y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)

print("r2 : ", r2)

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplot()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplot()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()

'''
[1999]  validation_0-logloss:-803.53387 validation_0-rmse:0.09501       validation_1-logloss:-752.75568 validation_1-rmse:2.52726
r2 :  0.9104977224000506
'''


