# male, female
# 넘파이로 저장
# fit_generator로 코딩
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
np.random.seed(26)


# # 이미지에 대한 생성옵션 정하기
# # 1. 증폭기능
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True, 
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=5,
                                   zoom_range=1.2,
                                   shear_range=0.7,
                                   fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)


# flow 또는 flow_from_directory
# 파일은 flow 폴더는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업.
train_generator = train_datagen.flow_from_directory(
    "./data/data2",
    target_size=(150,150),
    batch_size=5, # image를 5장씩
    class_mode="binary"
    # ,save_to_dir="./data/data1_2/train"
)
test_generator = test_datagen.flow_from_directory(
    "./data/data2",
    target_size=(150,150),
    batch_size=5, # image를 5장씩
    class_mode="binary"
    # ,save_to_dir="./data/data1_2/test"
)

model = Sequential()
model.add(Conv2D(10,(4,4),input_shape=(150,150,3))) 
model.add(Conv2D(10,(3,3)))                      
model.add(Conv2D(10,(3,3)))                                      
model.add(Conv2D(10,(2,2)))                            
model.add(MaxPooling2D(pool_size=2))   
model.add(Flatten())                                             
model.add(Dense(10,activation='relu'))                           
model.add(Dense(1,activation="sigmoid"))                       
model.summary()


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
hist = model.fit_generator(
    train_generator,
    steps_per_epoch = 20,
    epochs           = 100,
    validation_data  = test_generator,
    validation_steps = 4
)

loss     = hist.history["loss"]
val_loss = hist.history["val_loss"]
acc      = hist.history["acc"]
val_acc  = hist.history["val_acc"]


# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 단위 무엇인지 찾아볼것
plt.subplot(2,1,1)         # 2행 1열 중 첫번째
plt.plot(loss,marker='.',c='red',label='loss')
plt.plot(val_loss,marker='.',c='blue',label='val_loss')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)         # 2행 1열 중 두번째
plt.plot(acc,marker='.',c='red')
plt.plot(val_acc,marker='.',c='blue')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc']) # 라벨의 위치를 명시해주지 않으면 알아서 빈곳에 노출한다.

plt.show()


print("===============================================")

len = len(train_generator)
print(len) # 348

train_generator = train_datagen.flow_from_directory(
    "./data/data2",
    target_size=(150,150),
    batch_size=5*len, # image를 5장씩
    class_mode="binary"
    # ,save_to_dir="./data/data1_2/train"
)
test_generator = test_datagen.flow_from_directory(
    "./data/data2",
    target_size=(150,150),
    batch_size=5*len, # image를 5장씩
    class_mode="binary"
    # ,save_to_dir="./data/data1_2/test"
)

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save("./data/keras64_imageDataGenerator1_x_train",arr=train_generator[0][0])
np.save("./data/keras64_imageDataGenerator1_y_train",arr=train_generator[0][1])
np.save("./data/keras64_imageDataGenerator1_x_test",arr=test_generator[0][0])
np.save("./data/keras64_imageDataGenerator1_y_test",arr=test_generator[0][1])




# x_train = np.load("./data/keras63_imageDataGenerator2_x_train.npy")
# x_test  = np.load("./data/keras63_imageDataGenerator2_y_train.npy")
# y_train = np.load("./data/keras63_imageDataGenerator2_x_test.npy")
# y_test  = np.load("./data/keras63_imageDataGenerator2_y_test.npy")
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)






