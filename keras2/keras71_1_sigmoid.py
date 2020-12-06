import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

# print("x :", x )
# print("y:", y )
plt.plot(x,y)
plt.grid()
plt.show()

#계산이 끝난후 엑티베이션으로 넘어가서 다음 레이어에 영향을 준다