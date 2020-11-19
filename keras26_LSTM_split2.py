

#실습: 모델 구성
#train, test 분리하기 + early_stopping + validation_split
#predict


import numpy as np

#1. 데이터

dataset = np.array(range(1,101))
size = 5

#데이터 전처리 
def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]

        #aaa.append 줄일 수 있음
        #소스는 간결할수록 좋다
        # aaa.append([item for item in subset])
        aaa.append(subset)
          
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, size)

print(dataset)

dataset[:, 0:4]
dataset[:, 4]
#shape 확인하고 print한 다음 주석으로 적어 두기 

'''
x = dataset[0:100, 0:4]
y = dataset[0:100, 4]

x = x.reshape(x.sehape[0], 4, 1)

# 차원과는 관꼐없이 비례에 맞춰서 잘라준다!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split()
    x, y, train_size= 0.7, shuffle=True
)
'''