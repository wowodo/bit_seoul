#imageDataGenerator 이미지를 땡겨다 데이터화 해준다 라벨링도 해준다

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    class_mode='binary'     
)
xy_test = test_datagen.flow_from_directory(
 './data/data1/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'  

)

model.fit_generator(
    xy_train, #x-trian y_train
    step_per_epoch=100,
    epochs=20,
    validation_data = xy_test, validation_steps=4

)


