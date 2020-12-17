#넘파이 불러와서
#fit. 으로 코딩
import numpy as np
import matplotlib.pyplot as plt

x_train = np.load("./data/keras64_imageDataGenerator1_x_train.npy")
x_test = np.load("./data/keras64_imageDataGenerator1_x_test.npy")
y_train = np.load("./data/keras64_imageDataGenerator1_y_train.npy")
y_test = np.load("./data/keras64_imageDataGenerator1_y_test.npy")


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(Conv2D(50,(4,4),input_shape=(150,150,3))) 
model.add(Conv2D(80,(3,3),activation='relu'))                      
model.add(Conv2D(50,(3,3),activation='relu'))                                      
model.add(Conv2D(30,(2,2),activation='relu'))                            
model.add(MaxPooling2D(pool_size=2))   
model.add(Flatten())                                             
model.add(Dense(15,activation='relu'))                           
model.add(Dense(12,activation='relu'))                           
model.add(Dense(1,activation="sigmoid"))                       
model.summary()
# 3.
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=40, mode='auto')

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

hist=model.fit(x_train,y_train, epochs=1000,batch_size=32,verbose=1,validation_split=0.3,callbacks=[es])

loss,accuracy = model.evaluate(x_test,y_test,batch_size=16)
print("loss :",loss)
print("accuracy :",accuracy)


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
