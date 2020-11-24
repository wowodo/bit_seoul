

import numpy as np

dataset = np.array(range(1, 10))
# dataset = np.array([range(11), range(100, 200), range(1000, 2000)])

dataset = np.array([range(101, 201), range(311, 411), range(100)])

size = 6

#data가 2차원일 경우
def split_x(dataset, size):
    aaa = [] #는 테스트
    seq = dataset.shape[1]

    for i in range(seq-size+1):
        subset = dataset[:, i:(i+size)]

        aaa.append(subset)
        
        
        
    # print(type(aaa))
    return np.array(aaa)


dataset = split_x(dataset, size)
print("=============")
print(dataset)