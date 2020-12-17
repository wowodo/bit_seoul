'''
filters : 정수, 출력 공간의 차원 (예 : 컨볼 루션의 출력 필터 수).

kernel_size : 2D 컨볼 루션 창의 높이와 너비를 지정하는 정수 또는 2 개 정수의 튜플 / 목록입니다. 모든 공간 차원에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다.

strides : 높이와 너비를 따라 컨볼 루션의 스트라이드를 지정하는 정수 또는 2 개의 정수로 구성된 튜플 / 목록입니다. 모든 공간 차원에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다.
   stride 값! = 1을 지정하는 것은 임의의 dilation_rate값! = 1 을 지정하는 것과 호환되지 않습니다 .

padding : "valid"or 중 하나 "same"(대소 문자 구분 안함). "valid"패딩이 없음을 의미합니다. "same"출력이 입력과 동일한 높이 / 너비 치수를 갖도록 입력의 왼쪽 / 오른쪽 또는 위 / 아래에 균일하게 패딩됩니다.


filters : 던져주는 로드의 갯수  10
kernel_size : (2, 2)
strides : 2x2 자른걸 몇칸씩 옴길지 디폴트 1
padding 
input_shepe =  (rows ,cols, channels)
입력 모양: batch_size, row, cols, channels

참고  LSTM
units : 로드의 갯수
양의 정수, 출력 공간의 차원.
활성화 : 사용할 활성화 기능. 기본값 : 쌍곡 탄젠트 ( tanh). 을 통과 None하면 활성화가 적용되지 않습니다 
(예 : "선형"활성화 :) a(x) = x.

return_sequence ;부울. 마지막 출력을 반환할지 여부입니다. 
출력 시퀀스 또는 전체 시퀀스에서. 기본값 : False.

입력 모양: batch_size, timesteps(10일치씩 자른다 시간의 간격의 규칙), feature (몇개씩 자르는지)

입력 : 모양이있는 3D 텐서 [batch, timesteps, feature].
mask : [batch, timesteps]주어진 타임 스텝이 마스킹되어야하는지 여부를 나타내는 이진 텐서 (선택 사항, 기본값은 None).

training : 레이어가 학습 모드에서 동작해야하는지 추론 모드에서 동작해야하는지 나타내는 Python 부울입니다. 이 인수는 호출시 셀로 전달됩니다. 
dropout또는 recurrent_dropout이 사용되는 경우에만 관련 됩니다 (선택 사항, 기본값은 None).

initial_state : 셀의 첫 번째 호출에 전달할 초기 상태 텐서 목록 (선택 사항, 기본값 None은 0으로 채워진 초기 상태 텐서를 생성 하도록 함 ).

input_shape = (timesteps, feature)
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D
from tensorflow.keras.layers import Flatten 
#데이터를 일렬로 쫙 핀다

model = Sequential()
      # (2,2)가로세로  2x2자르기
      #conv 에서 출력은 그대로 전달된다 차원이 줄지 않는다 
model.add(Conv2D(10,(2,2), input_shape=(10,10,1))) # 9, 9, 10 흑백은 1 컬러는 3   conv  를 통과 하면 10 - 2 + 1 = 9 , 10이 된다  
                    #kernel_size
model.add(Conv2D(5,(2,2), padding='same'))         # 9, 9, 5  패딩을 same 인풋 아웃풋 똑같다
model.add(Conv2D(3,(3,3), padding='valid'))        # 7, 7, 3  패딩을 valid 를 쓰면 :  10 - 2 + 1 = 9 , 이공식을 사용하고
model.add(Conv2D(7,(2,2)))                         # 6, 6, 7  valid는 디폴트 
model.add(MaxPooling2D())                          # 3, 3, 7 맥스 풀링 통과하면 반으로 \
model.add(Flatten()) #데이터를 일렬로 쫙 핀다        # 3*3*7 = 63
model.add(Dense(1))#최종 아웃풋

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 5)           205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 3)           138
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 6, 7)           91
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 3, 3, 7)           0
_________________________________________________________________
flatten (Flatten)            (None, 63)                0
_________________________________________________________________
dense (Dense)                (None, 1)                 64
=================================================================
Total params: 548
Trainable params: 548
Non-trainable params: 0
_________________________________________________________________
'''



#레이어가 늘어나면서 중복되고 쓸모없는 데이터가 늘어 난다
#데이터의 크기가 큰게 피쳐가 높다
#Maxpauling2D를 쓰면 

'''
conv 레이어는 쓰면 쓸수록 성능이 좋아 진다
filters : 던져주는 로드의 갯수  10
kernel_size : (2, 2)
strides : 2x2 자른걸 몇칸씩 옴길지 디폴트 1
padding 
input_shepe =  (rows ,cols, channels)
입력 모양: batch_size(분석하는 몇장씩 작업을 할 것인지 (행을 전체에서 얼마나 자를지), row, cols, channels
'''