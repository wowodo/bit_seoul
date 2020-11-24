import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0, sep=',')
print(datasets)

print(datasets.shape)#150, 5

#index_col = None, 0, 1 / header = None, 0 ,1
'''
datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=1, sep=',')
              Unnamed: 0  sepal_width  petal_length  petal_width  species
sepal_length
5.1                    1          3.5           1.4          0.2        0
4.9                    2          3.0           1.4          0.2        0
4.7                    3          3.2           1.3          0.2        0
4.6                    4          3.1           1.5          0.2        0
5.0                    5          3.6           1.4          0.2        0
...                  ...          ...           ...          ...      ...
6.7                  146          3.0           5.2          2.3        2
6.3                  147          2.5           5.0          1.9        2
6.5                  148          3.0           5.2          2.0        2
6.2                  149          3.4           5.4          2.3        2
5.9                  150          3.0           5.1          1.8        2

[150 rows x 5 columns]
(150, 5)


datasets = pd.read_csv('./data/csv/iris_ys.csv', header=1, index_col=0, sep=',')
     5.1  3.5  1.4  0.2  0
1
2    4.9  3.0  1.4  0.2  0
3    4.7  3.2  1.3  0.2  0
4    4.6  3.1  1.5  0.2  0
5    5.0  3.6  1.4  0.2  0
6    5.4  3.9  1.7  0.4  0
..   ...  ...  ...  ... ..
146  6.7  3.0  5.2  2.3  2
147  6.3  2.5  5.0  1.9  2
148  6.5  3.0  5.2  2.0  2
149  6.2  3.4  5.4  2.3  2
150  5.9  3.0  5.1  1.8  2

[149 rows x 5 columns]
(149, 5)



datasets = pd.read_csv('./data/csv/iris_ys.csv', header=1, index_col=1, sep=',')
       1  3.5  1.4  0.2  0
5.1
4.9    2  3.0  1.4  0.2  0
4.7    3  3.2  1.3  0.2  0
4.6    4  3.1  1.5  0.2  0
5.0    5  3.6  1.4  0.2  0
5.4    6  3.9  1.7  0.4  0
..   ...  ...  ...  ... ..
6.7  146  3.0  5.2  2.3  2
6.3  147  2.5  5.0  1.9  2
6.5  148  3.0  5.2  2.0  2
6.2  149  3.4  5.4  2.3  2
5.9  150  3.0  5.1  1.8  2

[149 rows x 5 columns]
(149, 5)

datasets = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0, sep=',')
     sepal_length  sepal_width  petal_length  petal_width  species
1             5.1          3.5           1.4          0.2        0
2             4.9          3.0           1.4          0.2        0
3             4.7          3.2           1.3          0.2        0
4             4.6          3.1           1.5          0.2        0
5             5.0          3.6           1.4          0.2        0
..            ...          ...           ...          ...      ...
146           6.7          3.0           5.2          2.3        2
147           6.3          2.5           5.0          1.9        2
148           6.5          3.0           5.2          2.0        2
149           6.2          3.4           5.4          2.3        2
150           5.9          3.0           5.1          1.8        2

[150 rows x 5 columns]
(150, 5)

'''
print(datasets.head())
print(datasets.tail())
print(type(datasets))

#numpy  한가지 데이터 형만 써야 한다 데이터 형태가 통일 되어야 한다

#pandas  여러가지 데이터를 불러 올 수 있다

#aaa = #datasets를 넘파이로 바꿀것
# dataset = datasets.to_numpy()
aaa = datasets.values
print(type(datasets))
print(aaa.shape)

np.save('./data/iris_ys_pd.npy', arr=aaa)