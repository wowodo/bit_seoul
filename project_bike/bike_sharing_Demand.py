import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # KNeighbors 회기 Regressor만 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['axes.unicode_minus'] =False



'''
데이터 필드
날짜 - 시간당 날짜 + 타임 스탬프  
시즌 - 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울 
휴가 - 그 날이 휴일로 간주 여부
의 workingday 날이 둘 주말이나 휴일입니다 여부 -
날씨 - 1 : 클리어, 약간 구름, 부분적으로 흐림, 부분적으로 흐림
2 : 안개 + 흐림, 안개 + 부서진 구름, 안개 + 약간의 구름, 안개
3 : 약한 눈, 약한 비 + 뇌우 + 흩어진 구름, 약한 비 + 흩어진 구름
4 : 폭우 + 얼음 깔판 + 뇌우 + 미스트, 눈 + 안개 
온도 - 섭씨 온도
atemp - 섭씨 온도 "느낌"
습도 - 상대 습도
풍속- 풍속
캐주얼 - 비 등록 사용자 대여 횟수 개시
등록 - 등록 된 사용자의 번호 렌털 개시
카운트 - 렌털 총 개수
'''

train = pd.read_csv("./data/csv/train.csv", parse_dates=["datetime"])
print(train.shape) #(10886, 12)
test = pd.read_csv("./data/csv/test.csv", parse_dates=["datetime"])
print(test.shape) #(6493, 9)


#Feature Engineering
#parse_dates=["datetime"] 이 데이트 타임을 세분화 해서 나눈다
train["year"] = train ["datetime"].dt.year
train["month"] = train ["datetime"].dt.month
train["day"] = train ["datetime"].dt.day
train["hour"] = train ["datetime"].dt.hour
train["minute"] = train ["datetime"].dt.minute
train["second"] = train ["datetime"].dt.second
train["dayofweek"] = train ["datetime"].dt.dayofweek
print(train.shape) #(10886, 19)

test["year"] = test ["datetime"].dt.year
test["month"] = test ["datetime"].dt.month
test["day"] = test ["datetime"].dt.day
test["hour"] = test ["datetime"].dt.hour
test["minute"] =test ["datetime"].dt.minute
test["second"] = test ["datetime"].dt.second
test["dayofweek"] = test ["datetime"].dt.dayofweek
print(test.shape) #(6493, 16)

fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count',title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='Count',title="test windspeed")
sns.countplot(data=test, x="windspeed", ax=axes[1])

# plt.show()


# 풍속이 0인것과 아닌것의 세트를 나누어 준다
trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]
print(trainWind0)  #[1313 rows x 19 columns]
print(trainWindNot0) #[9573 rows x 19 columns]

def predict_windspeed(data):

    
    # 풍속이 0인것과 아닌것의 세트를 나누어 준다
    dataWind0 = train.loc[data['windspeed'] == 0]
    dataWindNot0 = train.loc[data['windspeed'] != 0]

    #풍속을 예측할 피쳐를 선택한다
    wCol = ["season", "weather", "humidity", "month", "temp","year","atemp"]

    #풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다.
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")
    #랜덤포레스트 분류기를 사용한다.
    rfModel_wind = RandomForestClassifier()
    #wCol에 있는 피처의 값을 바탕으로 풍속을 학습시킨다.
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])
    #학습한 값을 바탕으로 풍속이 0으로 기록 된 데이터의 풍속을 예측한다.
    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])
    #값을 다 예측 후 비교해 보기 위해
    #예측한 값을 넣어 줄 데이터 프레임을 새로 만든다
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0

    #값이 0으로 기록 된 풍속에 대해 예측한 값을 넣어준다.
    predictWind0["windspeed"] = wind0Values

    #dataWindNot0 0이 아닌 풍속이 있는 데이터 프레임에 예측한 값이 있는 데이터 프레임을 합쳐준다
    data = predictWindNot0.append(predictWind0)

    #풍속의 데이터타입을   float 으로 지정해 준다.
    data["windspeed"] = data["windspeed"].astype("float")

    data.reset_index(inplace=True)
    data.drop('index',inplace=True, axis=1)

    return data

#0값을 조정한다
train = predict_windspeed(train)

#widspeed 의 0값을 조정한 데이터를 시각화
fig, ax1 = plt.subplots()
fig.set_size_inches(18,6)

plt.sca(ax1)
plt.xticks(rotation=30, ha='right')
ax1.set(ylabel='Count',title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=ax1)

# plt.show()

'''
Feature Selection

-신호와 잡음을 구분해야 한다.
-피처가 많다고 해서 무조건 좋은 성능을 내지 않는다
-피처를 하나씩 추가하고 변경해 가면서 성능이 좋지 않은 피처는 제거하도록한다

'''

#연속형  feature (온도, 습도, 풍속 숫자에 따라서 강약을 볼수있다) 와 범주형  feature(시즌, 웨더, 요일  1 ,2, 3, 4 숫자를 곱한다고해서 달라지지 않는다)
#연속형  feature = ["temp", "humidity", "windspeed","atemp"]
#범주형  feature의  type을  category로 변경 해 준다

categorical_feature_name =["season","holiday","workingday","weather",
                            "dayofweek","month","year","hour"]

for val in categorical_feature_name:
    train[val] = train[val].astype("category")
    test[val] = test[val].astype("category")

feature_names = ["season","weather","temp","atemp","humidity","windspeed",
                    "year", "hour","dayofweek","holiday","workingday"]

feature_names
#x_train 데이터 셋 만들기
X_train = train[feature_names]

print(X_train.shape)
X_train.head()

'''
                 datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count  year  month  day  hour  minute  second  dayofweek
0     2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81        0.0       3          13     16  2011      1    1     0       0       0          5
1     2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80        0.0       8          32     40  2011      1    1     1       0       0          5
2     2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80        0.0       5          27     32  2011      1    1     2       0       0          5
3     2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75        0.0       3          10     13  2011      1    1     3       0       0          5
4     2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75        0.0       0           1      1  2011      1    1     4       0       0          5
...                   ...     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...    ...   ...    ...  ...   ...     ...     ...        ...
10826 2012-12-17 12:00:00       4        0           1        2  16.40  20.455        87        0.0      21         211    232  2012     12   17    12       0       0          0
10829 2012-12-17 15:00:00       4        0           1        2  17.22  21.210        88        0.0      15         196    211  2012     12   17    15       0       0          0
10846 2012-12-18 08:00:00       4        0           1        1  15.58  19.695        94        0.0      10         652    662  2012     12   18     8       0       0          1
10860 2012-12-18 22:00:00       4        0           1        1  13.94  16.665        49        0.0       5         127    132  2012     12   18    22       0       0          1
10862 2012-12-19 00:00:00       4        0           1        1  12.30  15.910        61        0.0       6          35     41  2012     12   19     0       0       0          2
[1313 rows x 19 columns]
'''
#x_test 데이터 셋 만들기
X_test = test[feature_names]

print(X_test.shape)
X_test.head()
'''
                 datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count  year  month  day  hour  minute  second  dayofweek
5     2011-01-01 05:00:00       1        0           0        2   9.84  12.880        75     6.0032       0           1      1  2011      1    1     5       0       0          5
10    2011-01-01 10:00:00       1        0           0        1  15.58  19.695        76    16.9979      12          24     36  2011      1    1    10       0       0          5
11    2011-01-01 11:00:00       1        0           0        1  14.76  16.665        81    19.0012      26          30     56  2011      1    1    11       0       0          5
12    2011-01-01 12:00:00       1        0           0        1  17.22  21.210        77    19.0012      29          55     84  2011      1    1    12       0       0          5
13    2011-01-01 13:00:00       1        0           0        2  18.86  22.725        72    19.9995      47          47     94  2011      1    1    13       0       0          5
...                   ...     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...    ...   ...    ...  ...   ...     ...     ...        ...
10881 2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329    336  2012     12   19    19       0       0          2
10882 2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231    241  2012     12   19    20       0       0          2
10883 2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164    168  2012     12   19    21       0       0          2
10884 2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117    129  2012     12   19    22       0       0          2
10885 2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84     88  2012     12   19    23       0       0          2

[9573 rows x 19 columns]
'''

label_name = "count"
y_train = train[label_name]

print(y_train.shape)  #(10886,) 카운트 값을 예측한다
y_train.head()


#RNSE 검사 하는 방법
from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values):
    #넘파이로 배열 형태로 바꿔준다
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    #예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values +1) 
    log_actual = np.log(actual_values + 1)

    #위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다
    difference = log_predict - log_actual
    #difference = (log_predict - log_actual)**2
    difference = np.square(difference)

    #평균을 낸다
    mean_difference = difference.mean()

    #다시 루트를 씌운다
    score = np.sqrt(mean_difference)

    return score

    rmsle_scorer = make_scorer(rmsle)
    

    #kFold 교차검증
        #데이터를 폴드라 부르는 비슷한 크기의 부분집합(n_splits)으로 나누고 각각의 폴드 정확도를 측정한다
        #첫번 째 폴드를 테스트 세트로 사용하고 나머지 폴드를 훈련세트로 사용하여 학습한다.
        #나머지 훈련세트로 만들어진 세트의 정확도를 첫 번째 폴드로 평가한다.
        #다음은 두 번째 폴드가 테스트 세트가 되고 나머지 폴드의 훈련세트를 두 번째 폴드로 정확도를 측정한다
        #이과정을 마지막 폴드까지 반복한다
        #이렇게 훈련세트와 테스트세트로 나누는 N개의 분할마다 정확도를 측정하여 평균 값을 낸게 정확도가 된다.


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#RandomForest

from sklearn.ensemble import RandomForestRegressor

max_depth_list =[]

model = RandomForestRegressor(n_estimators=100,
                                n_jobs=-1,
                                random_state=0)
model

# score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
# score = score.mean()
# #0에 근접할수록 좋은 데이터
# print("Score={0:.5f".format(score))#타임으로 스코어를 찍는다

#Train
model.fit(X_train, y_train)

#예측
predictions = model.predict(X_test)

print(predictions.shape)
predictions[:10]

# 예측한 데이터를 시각화 해본다. 
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
ax1.set(title="train")
sns.distplot(predictions,ax=ax2,bins=50)
ax2.set(title="test")
plt.show()

submission = pd.read_csv("data/csv/sampleSubmission.csv")
submission

submission["count"] = predictions

print(submission.shape)
print(submission.head())
'''
              datetime  count
0  2011-01-20 00:00:00  12.16
1  2011-01-20 01:00:00   5.02
2  2011-01-20 02:00:00   4.32
3  2011-01-20 03:00:00   3.42
4  2011-01-20 04:00:00   3.18
'''

