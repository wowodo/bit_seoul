import numpy as np
import matplotlib.pyplot as plt
import librosa
import timeit


# import timeit
#이 모듈은 파이썬 코드의 작은 조각의 시간을 측정하는 간단한 방법을 제공합니다. 
# 명령 줄 인터페이스뿐만 아니라 콜러블도 있습니다. 실행 시간을 측정에 따르는 
# 흔한 함정들을 피할 수 있습니다. O’Reilly가 출판한 Python Cookbook에 있는 
# Tim Peters의 《Algorithms》 장의 개요도 참조하십시오.
file_path = 'audio/nsynth-test/audio_path/bass_electronic_018-022-100.wav'
start_time = timeit.default_timer() # 시작 시간 체크
y, sr =librosa.load(file_path)
# y :진동파의 폭을 의미한다
#sr : sampling rate 또는 sampling frequncy(샘플링 주파수)를 뜻한다 sr = 22050 대상 샘플링 속도
'''
이산적 신호를 만들기 위해 연속적 심호에서 얻어진 단위 시간 초당 샘플링 횟수를 정의한다
'''

fft = np.fft.fft(y)/len(y) #wav 값을 python에 맞게 변경
magnitude = np.abs(fft)#이상치 값을  np.abssolute 절대 값으로 만들어서 magnitude 에 넣어준다

#np.linspace(fft) 절대 값을 취하는 과정을 거쳐 스펙트로그램을 만들었기 때문에 역변환이 불가능
f = np.linspace(0, sr, len(magnitude))

left_spectrum = magnitude[:int(len(magnitude)/2)]#절반 잘라서 대칭 안되게 하기
left_f = f[:int(len(magnitude)/2)]

pitch_index = np.where((left_f>20.0)&(left_f<4200.0)) #20.0 ~ 4200.0 헤르츠의 index 구함
print('pitch_index : ',pitch_index)
pitch_freq = left_f[pitch_index]
pitch_mag = left_spectrum[pitch_index]