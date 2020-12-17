#나머지 6개를 저장하시오
#코드를 완성하시오

# from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# np.save('./data/fashion_x_train.npy', arr=x_train) #x_train이라는 numpy가 mnist_x_train.npy로 저장된다 
# np.save('./data/fashion_y_train.npy', arr=y_train)

# np.save('./data/fashion_x_test.npy', arr=x_test)
# np.save('./data/fashion_y_test.npy', arr=y_test)

# from tensorflow.keras.datasets import cifar10#1. 데이터
# (x_train, y_train), (x_test, y_test) = cifar10.load_data() #괄호 주의

# np.save('./data/cifar10_x_train.npy', arr=x_train) #x_train이라는 numpy가 mnist_x_train.npy로 저장된다 
# np.save('./data/cifar10_y_train.npy', arr=y_train)

# np.save('./data/cifar10_x_test.npy', arr=x_test)
# np.save('./data/cifar10_y_test.npy', arr=y_test)

# from tensorflow.keras.datasets import cifar100
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# np.save('./data/cifar100_x_train.npy', arr=x_train) #x_train이라는 numpy가 mnist_x_train.npy로 저장된다 
# np.save('./data/cifar100_y_train.npy', arr=y_train)

# np.save('./data/cifar100_x_test.npy', arr=x_test)
# np.save('./data/cifar100_y_test.npy', arr=y_test)

# from sklearn.datasets import load_boston
# dataset = load_boston()
# x_data = dataset.data
# y_data = dataset.target


# np.save('./data/boston_x.npy', arr=x_data)
# np.save('./data/boston_y.npy', arr=y_data)

#1. 데이터
# from sklearn.datasets import load_diabetes
# dataset = load_diabetes() #data(X)와 target(Y)으로 구분되어 있다
# x_data = dataset.data
# y_data = dataset.target

# np.save('./data/diabetes_x.npy', arr=x_data)
# np.save('./data/diabetes_y.npy', arr=y_data)

# from sklearn.datasets import load_breast_cancer

# dataset = load_breast_cancer()
# x_data = dataset.data
# y_data = dataset.target

# np.save('./data/cancer_x.npy', arr=x_data)
# np.save('./data/cancer_y.npy', arr=y_data)