
# 한 레이어서 일어나는 일 러닝메이트 옵티마이져

weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 0.001 #0.001 디폴트 #0.1 / 1 /0.0001 / 10

for iteration in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction) **2

    print("Error : " + str(error) + "\tprediction : " + str(prediction))

    up_prediction = input *(weight + lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input *(weight + lr)
    down_error = (goal_prediction - up_prediction) **2

    if(down_error < up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight - lr


'''
lr = 0.001

 0.30250000000000005     prediction : 0.25
--------------------

lr = 0.0001

 0.30250000000000005     prediction : 0.25

lr = 1
 0.30250000000000005     prediction : 0.25
'''