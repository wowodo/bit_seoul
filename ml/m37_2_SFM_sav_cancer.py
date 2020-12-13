
# SFM에 save, load 를 적용해서
#저장파일은 제일 좋은 값만 남기고 다 지울것


# 이벨류에이트 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target
# 1.데이터
x, y  = load_breast_cancer(return_X_y=True)

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
model.fit(x_train, y_train, verbose=True, eval_metric='rmse',    #메트릭스는 평가라 영향을 주지 않는다 rmse mse 에 루트를 씌운것
                            eval_set=[(x_train, y_train),(x_test, y_test)]  #훈련셋과 테스트 벌보스 둘다 볼수 있다 

)
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
import pickle
thresholds = np.sort(model.feature_importances_) #그 모델에 대한 컬럼 중요도를 뽑아서 sort (정렬한다) 
print(thresholds)

# 우리가 쓰는 컴퓨터 코어 6개 12쓰레드 
import pickle
n     = 0
b_acc = 0

#selectfromModel

for thresh in thresholds: #thresholds에서 하나씩 뽑아서 thresh 널어 준다
    selection = SelectFromModel(model, threshold=thresh, prefit=True)#SelectFromModel에 모델과 컬럼 중요도를 넣어주면
    #낮은 순으로 들어가 있는 것을 model 에서 하나씩 빼서 셀렉션으로 넣어 준다

    select_x_train = selection.transform(x_train) #x_train은 자르고 남은 수가 된다. 그것을 slect_x_train으로 만들어 준다
    selection_model = XGBClassifier()#새로운 모델을 생성한다
    selection_model.fit(select_x_train,y_train)#남은 컬럼으로 모델 핏을 해본다

    select_x_test = selection.transform(x_test)#위에 x_train 과 같이 자르고 남을것을 select_x_test에 넣어준다
    y_predict = selection_model.predict(select_x_test)# select_x_test 으로 바꾼 값을 예측 해본다.

    acc = accuracy_score(select_x_test,y_test) #select_x_test와 y_test의 스코어를 매긴다
    if acc > b_acc:
        n = select_x_train.shape[1]
        b_acc = acc
        L_selection = selection
        pickle.dump(model, open("./save/xgb_save/cancser.pickle.dat", "wb"))
        print("Thresh=%.3f, n=%d, acc: %.15f%%"%(thresh,select_x_train.shape[1],acc))
     
       
