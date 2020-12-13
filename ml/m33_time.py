from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=66,shuffle=True)

model = XGBRegressor(n_jobs=-1) # 엑스지 부스터로  n_jobs=-1 씨피유 모든코어 다쓴다 
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("r2 score : ",score)  #R2값을 뽑는다

thresholds = np.sort(model.feature_importances_) #그 모델에 대한 컬럼 중요도를 뽑아서 sort (정렬한다) 
print(thresholds)

# 우리가 쓰는 컴퓨터 코어 6개 12쓰레드 

import time
start1 = time.time()
for thresh in thresholds: # thresh 가 thresholds의 갯수 만큼 돈
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100.0)) #시간 측정할때에는 보여주는것도 시간에 잡힌다

start2 = time.time()
for thresh in thresholds: # thresh 가 thresholds의 갯수 만큼 돈
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=8)  #n_jobs = -1 이 제일 좋은것 같지만 아니다
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    # print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100.0))   #시간 측정할때에는 보여주는것도 시간에 잡힌다

end = start2 - start1
print("그냥 걸린 시간 : ", end)
end2 = time.time() - start2
print("잡스 걸린 시간 : ", end2) #n_jobs=-1 코어 


