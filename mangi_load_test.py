import numpy as np
import matplotlib.pyplot as plt

x = np.load('./npy/all_scale_x.npy')
y = np.load('./npy/all_scale_y.npy')

print(x)
print(y)


print(x.shape)#(365, 12)
print(y.shape)#(365,)
tmp_x = np.unique(y)
print(tmp_x)
plt.plot(tmp_x, x[1])
plt.show()