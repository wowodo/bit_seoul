#train 값으로 카운트를 자르고 y 값으로 사용
import numpy as np
import pandas as pd

#판다스로  파일 읽어 오기
bike = pd.read_csv('./data/csv/train.csv',header=0,index_col=0,sep=",")
bike_test = pd.read_csv('./data/csv/test.csv',header=0,index_col=0,sep=",")

bike_sample = pd.read_csv('./data/csv/sampleSubmission.csv',header=0,index_col=0,sep=",")



# print(bike) #[10886 rows x 12 columns]

#칼럼 뽑아 내기 x_y 나눠서
bike_x = bike [["season","holiday","workingday","temp","weather","atemp","windspeed","humidity"]]
bike_y = bike[["count"]]
print(bike)

#오름차순으로 변경
# bike = bike.sort_values(by=['datetime'], ascending=['True'])

# print(bike)

bike = bike.values
bike_test = bike_test.values
bike_sample = bike_sample.values

print(type(bike)) #[10886 rows x 7 columns]

#넘파이로 저장
# np.save('./project/bike_x.npy', arr=bike_x)
# np.save('./project/bike_y.npy', arr=bike_y)

np.save('./project/bike_x.npy', arr=bike_x)
np.save('./project/bike_y.npy', arr=bike_y)

np.save('./project/bike_test_x.npy', arr=bike_test)   #테스트에는  y값 필요 없음


np.save('./project/bike_sample.npy', arr=bike_sample)   #테스트에는  y값 필요 없음



