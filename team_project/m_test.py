import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.models import Model

x = np.load('./npy/all_scale_x.npy')
y = np.load('./npy/all_scale_y.npy')

print(x.shape)# (365, 12)
print(y.shape)# (365,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66,shuffle=True)

print(x_train.shape, y_train.shape)

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("score",score)

'''
(365, 12)
(365,)
score 0.9727272727272728
'''