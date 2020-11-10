#1. 데이터 #1부터 100
import numpy as np
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array([range(101, 201), range(711, 811), range(100)])

print(x[1][10])
print(x.shape) #(3, )
print(np.array(x))

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
# (100, 3)
#MLP 멀티 레이어 퍼센트론