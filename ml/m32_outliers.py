import numpy as np


def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사분위 : ", quartile_1) #3.25
    print("3사분위 : ", quartile_3) # 97.25
    iqr = quartile_3 = quartile_1 #94.25
    lower_bound = quartile_1 = (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound ) | (data_out<lower_bound))
    #데이터 지검 1.5범위가 위로 아래로넘어 가는애들을 찾아서 리턴해라

a = np.array([1,2,3,4,10000,6,7,5000,90,100])

b = outliers(a)# 전체 기링에서 4/3지점 4/2지점을 찾는다  1부터 10000까지 에서 

print("이상치의 위치 : ", b)
