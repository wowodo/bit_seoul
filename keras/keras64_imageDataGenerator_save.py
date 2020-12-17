#imageDataGenerator 이미지를 땡겨다 데이터화 해준다 라벨링도 해준다

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
np.random.seed(33)

#이미지에 대한 생성 옵션 정하기
train_datagen = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    rotation_range=5,
                                    zoom_range=1.2,
                                    shear_range=0.7,
                                    fill_mode='nearest' #빈칸을 주변과 같게 한다                            
                                    
)
test_datagen = ImageDataGenerator(rescale=1./255)  #테스트는 기존이미지 가지고 한다 


#flow  또는 #flow_from_directory (폴더에서 땡겨온다)
#이미지 불러오기
#실제 데이터가 있는 곳을 알려주고 , 이미지를 불러오는 작업
#generator x y가 같이 들어가있다
xy_train = train_datagen.flow_from_directory(
    './data/data1/train',
    target_size=(150, 150),
    batch_size=5,               
    class_mode='binary',
    # save_to_dir='./data/data1_2/tarin'     #변환된 이미지를 seve를 한다  .jpg
)
xy_test = test_datagen.flow_from_directory(
 './data/data1/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'  

)

#numpy 로 변환해서 작업
print("-------------------------------------")
print(type(xy_train))
print(xy_train[0])
# print(xy_train[0].shape) #x와 y두종류가 들어가있다 error
print(type(xy_train[0][0]))   #  <class 'numpy.ndarray'>
print(xy_train[0][0].shape)   #(5, 150, 150, 3)
# print(xy_train[0][1].shape)   #(5,) 는 y 
# print(xy_train[1][1].shape)   #batch_size=5,
print(len(xy_train))          #32  160개를 batch_size=5 자르면 32
print("-------------------------------------")
print(xy_train[0][0][0])

print(xy_train[0][1][:5])

# np.save('./data/keras63_train_x.npy', arr=xy_train[0][0])
# np.save('./data/keras63_train_y.npy', arr=xy_train[0][1])
# np.save('./data/keras63_test_x.npy', arr=xy_test[0][0])
# np.save('./data/keras63_test_y.npy', arr=xy_test[0][1])

#모델
model = Sequential()
model(Conv2D(10, (3,3),padding='same', input_shape=(150,150,3)))
model.add(Conv2D(20, (2,2),padding='valid'))
model.add(Conv2D(30, (3,3)))
model.add(Conv2D(40,(2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='sigmoid'))




# 평가 예측
model.compile(loss='categorical_crossentropy', optimizer="adam",
                metrics=['accuracy']) #'mean_squared_error'
#                 #10개를 합친 값은 1이 되고 거기서 제일 큰걸

history = model.fit_generator(
    xy_train, #x-trian y_train
    step_per_epoch=100,
    epochs=20,
    validation_data = xy_test, validation_steps=4
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_acc = history.history['loss']

#시각화 완성
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #단위 무엇인지 찾아볼것!

plt.subplot(2,1,1) #(2행 1열에서 1번째 그림)
plt.plot(history.history['loss'], marker='.', c='red', label='loss')  # x값과, y값이 들어가야 합니다
plt.plot(history.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.show()