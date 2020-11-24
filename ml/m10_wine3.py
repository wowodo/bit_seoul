import pandas as pd
import numpy as np

wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0) 

count_data = wine.groupby('quality')['quality'].count() #퀄리트 안에 있는 객체 들을 세겠다

print(count_data)

import matplotlib.pyplot as plt
count_data.plot()
plt.show()
'''

5번 6번 7번 데이터가 너무 많아서 결과 값이 5,6,7번이 나올 확률이 높다
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''

