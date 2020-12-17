import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
#1~9까지 손글씨
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_train[0])  #28 행 28열 0은 빈칸
print(y_train[0])


plt.imshow(x_train[0],'gray')
plt.show()

#분류할때 공평한 수를 가져야 한다  0~9평등하다
#one-hot(원핫)인코딩이란? 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말한다.
            #즉, 1개만 Hot(True)이고 나머지는 Cold(False)이다.
            #예를들면 [0, 0, 0, 0, 1]이다. 5번째(Zero-based 인덱스이므로 4)만 1이고 나머지는 0이다.


#클래스 sklearn.preprocessing.OneHotEncoder( * , categories = 'auto' , drop = None , sparse = True , dtype = <class 'numpy.float64'> , handle_unknown = 'error' )
'''
범주 형 특성을 원-핫 숫자 형 배열로 인코딩합니다.

이 변환기에 대한 입력은 정수 또는 문자열의 배열과 유사해야하며 범주 형 (이산) 기능이 사용하는 값을 나타냅니다. 기능은 원-핫 (일명 'one-of-K'또는 '더미') 인코딩 체계를 사용하여 인코딩됩니다. 이렇게하면 각 범주에 대한 이진 열이 생성되고 희소 행렬 또는 조밀 배열이 반환됩니다 ( sparse 매개 변수 에 따라 다름 ).

기본적으로 인코더는 각 기능의 고유 한 값을 기반으로 범주를 파생합니다. 또는 categories 수동으로 지정할 수도 있습니다.

이 인코딩은 범주 형 데이터를 많은 scikit-learn 추정자, 특히 표준 커널을 사용하는 선형 모델 및 SVM에 공급하는 데 필요합니다.

참고 : y 레이블의 원-핫 인코딩은 대신 LabelBinarizer를 사용해야합니다.
'''