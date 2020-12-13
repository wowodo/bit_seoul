# 이벨류에이트 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target
# 1.데이터
x, y  = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    shuffle=True,
                                                    random_state=77
)

#2.모델
model = XGBClassifier(n_estimators=1200, learning_rate=0.01)
# model = XGBRegressor(learning_rate=0.01)

'''
n_estimators
디폴트로 100번돈다

n_estimators=10,
10번 훈련
'''

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric='mlogloss',    #메트릭스는 평가라 영향을 주지 않는다 rmse mse 에 루트를 씌운것
                            eval_set=[(x_train, y_train),(x_test, y_test)]  #훈련셋과 테스트 벌보스 둘다 볼수 있다 

)
# eval_metric의 대표 파람 =  rmse, mae, logloss, error, auc

#
results = model.evals_result()
# print("eval's results : ", results)

#4. 평가,예측


y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("accuracy : ", acc)

'''
[1199]  validation_0-mlogloss:0.01614   validation_1-mlogloss:0.45564
accuracy :  0.8333333333333334

eval_metric [목표에 따른 기본값]
유효성 검사 데이터에 대한 평가 메트릭, 기본 메트릭은 목표에 따라 할당됩니다
 (회귀의 경우 rmse, 분류의 경우 logloss, 순위의 평균 정밀도).

 mae: 절대 오류를 의미

mape: 절대 백분율 오류를 의미합니다.

mphe: 가짜 Huber 오류를 의미합니다 . reg:pseudohubererror목표의 기본 측정 항목입니다 .

logloss: 음의 로그 우도

error: 이진 분류 오류율. 로 계산됩니다 . 예측의 경우 평가는 예측 값이 0.5보다 큰 인스턴스를 긍정적 인 인스턴스로 간주하고 나머지는 부정적인 인스턴스로 간
'''

