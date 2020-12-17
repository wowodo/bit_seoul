#남이 불러오면 파일 명이 나온다.

import p11_car
import p12_tv

print("===========")
print("do.py 의  module 이름은 ",__name__)
print("-===========")

p11_car.drive() #.drive로 하면 그안에 있는것만 불러 온다.
p12_tv.watch()