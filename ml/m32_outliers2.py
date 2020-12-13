#퍼센타일
# np.where() : 비교후 인덱스 반환
# np.percentile() : data의 interpolation으로 데이터를 자른 value를 반환
#                 : 값들을 정렬해준다. 
# linear : 백분위 
# nearest : linear의 가장 가까운 진짜 값
# lower : 백분위보다 낮은 값 중 제일큰 값 
# higher : 백분위보다 큰값 중 제일낮은 값
# midpoint : linear로 나온 앞뒤값의 평균값.
#
import numpy as np


def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75], interpolation='linear')
    print("1사분위 : ", quartile_1) #3.25
    print("3사분위 : ", quartile_3) # 97.25
    iqr = quartile_3 = quartile_1 #94.25
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound ) | (data_out<lower_bound))
    #데이터 지검 1.5범위가 위로 아래로넘어 가는애들을 찾아서 리턴해라

a = np.array([[1,2,3,4,10000,6,7,5000,90,100],
            [1,2,3,4,10000,6,7,3000,90,100],
            [1,2,3,4,50,6,7,5000,90,100],
            [1,2,3,4,5,6,7,2000,90,100]]
)
a = a.transpose()
b = outliers(a)# 전체 기링에서 4/3지점 4/2지점을 찾는다  1부터 10000까지 에서 

print("이상치의 위치 : ", b)

#interpolation='linear'
# 이상치의 위치 :  (
#     array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=int64), 
#     array([4, 7, 8, 9, 4, 7, 8, 9, 4, 7, 8, 9, 7, 8, 9], dtype=int64))

#
