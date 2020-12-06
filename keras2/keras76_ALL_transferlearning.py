#전의 학습 (남의 잘한거 뺏는다 모델 가중치) 이미지 넷이라는 훈련시킨 것을 디폴트로 가지고 있다

##$$$$$$$$$$$$함수형으로 하는 방법있다 해봐라
#전의 학습
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile

from tensorflow.keras.layers import Dense, Flatten, BatchNormalization,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation

nasnetMobile = NASNetMobile() 

# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) #100X100 3 짜리 이미지를 받아서 훈련시키지 않겠다

nasnetMobile.trainable=True    #그냥 있는 그대로 VGG16에 있는 이미지넷에 있는 가중치로 쓰고    새로 훈련을 시키지 않겠다
                         #Trainable params: 0 

# model.trainable=Treu     #웨이트와 바이어스가 나온다
nasnetMobile.summary()

print("동결하기 전 훈련되는 가중치의 수",len(nasnetMobile.trainable_weights))

# #인풋 쉐이프가 고정 되있다
# model = Sequential()
# model.add(vgg16) 
# model.add(Flatten()) #평탄화 해준다
# model.add(Dense(256)) #가중치의 수 4
# # model.add(BatchNormalization()) #8 가중치 계산 된다
# # model.add(Dropout(0.2))# 가중치 연산 안됨
# model.add(Activation('relu'))  #통과

# model.add(Dense(256)) # 가중치의 수  6
# model.add(Dense(10, activation='softmax')) # add가 없다고 에러가 난다 vgg16 함수형인지 시퀀샬인지 모른다 그래서 from tensorflow.keras.models import Sequential

# model.summary()
#  마지막에 출력되는(5130) 부분에서 Dense(10, 5130 * 10   Trainable params: 5,130
# print("동결하기 전 훈련되는 가중치의 수",len(model.trainable_weights))
# print(model.trainable_weights)//

#과적합 줄이기 3 훈련을 늘린다, 베치노말제이션 정규화, 


#모델이 가장 순수했을때의 파라미터의 갯수와 가중치 수를 정리하시오
#ex) VGG16  : Trainable params: 138,357,544 
'''
vgg19
Trainable params: 138,357,544
동결하기 전 훈련되는 가중치의 수 32
---------------------------
xception 
Trainable params: 22,855,952
동결하기 전 훈련되는 가중치의 수 156
---------------------------
ResNet101
Trainable params: 44,601,832
동결하기 전 훈련되는 가중치의 수 418
------------------------------
resNet101V2
Trainable params: 44,577,896
동결하기 전 훈련되는 가중치의 수 344
----------------------------------
resNet152
Trainable params: 60,268,520
동결하기 전 훈련되는 가중치의 수 622
--------------------------------
resNet50
Trainable params: 25,583,592
동결하기 전 훈련되는 가중치의 수 214
---------------------------------
inceptionV3 
Trainable params: 23,817,352
동결하기 전 훈련되는 가중치의 수 190
^
---------------------------------
inceptionResNetV2
Trainable params: 55,813,192
동결하기 전 훈련되는 가중치의 수 490
----------------------------------
denseNet121
Trainable params: 7,978,856
동결하기 전 훈련되는 가중치의 수 364
-----------------------------------
mobileNetV2
Trainable params: 3,504,872
동결하기 전 훈련되는 가중치의 수 158
-----------------------------
nasnetMobile
Trainable params: 5,289,978
동결하기 전 훈련되는 가중치의 수 742
'''

#다 채워넣기