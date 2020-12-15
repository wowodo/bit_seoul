import numpy as np
import matplotlib.pyplot as plt
import librosa

file_path = 'audio/nsynth-test/audio_path/bass_electronic_018-022-100.wav'

y, sr = librosa.load(file_path) #파일_경로
# y는 amplitude 를 뜻하는데  이것은 진폭 주기적으로 진동하는 파의 진동 폭을 의미한다

# sr은 sampling rate 또는  sampling frequency (샘플링 주파수)를 뜻한다  sr = 22050 대상 샘플링 속도
'''
이산적 신호를 만들기 위해 연속적 신호에서 얻어진 단위 시간 초당 샘플링 횟수를 정의한다 단위는 Hz이다
(이산적 역속적이지 않고 끊겨져 있다는 얘기)
'''

#FFT(Fast Fourier Transform) 즉 고속 푸리에 변환이 되겠습니다
'''
계속 변화하는 데이터를 주파수 영역으로 가져다가 어떤 주파수 들이 사용되고 있는지를 알아볼 때 쓰게 됩니다.
'''
# magnitude - 크기 
fft = np.fft.fft(y) # wav 값을 python에 맞게 변경 
magnitude = np.absolute(fft) # 이상치 값을 를 np.abs 절대 값으로 변환해서 크기에 넣어 준다? - 절대 값은 음수를 양수로 바꿔준다
#보통 푸리에 변환은 역변환이 가능하기 때문에 바로 다시 음원으로 변형도 가능하지만
# np.absolute(fft) 절대값을 취하는 과정을 거쳐서 스펙트로그램을 만들었기 때문에 역변환이 불가능합니다.

#np.linspace - (시작, 정지, num=50, 끝점=True, retstep=거짓, dtype=없음, 축=0)
#지정된 간격으로 균등하게 간격이 있는 숫자를 반환합니다. 간격을 통해 계산된 num [시작, 중지]를반환합니다. 간격의 끝점을 선택적으로 제외할 수 있습니다.
f = np.linspace(0, sr, len(magnitude)) #몇개의 일정한 간격으로 요소를 만들기 /규모 크기


# 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날리고 앞쪽 절반만 사용한다.
left_spectrum = magnitude[:int(len(magnitude)/2)] # 절반 잘라서 대칭안되게 하기
left_f = f[:int(len(magnitude)/2)]

plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency") #x쪽 라벨에 
plt.ylabel("Magnitude")#t 쪾 라벨에서 
plt.title("Power spectrum")#맨 위쪽 타이틀
plt.show()


#numpy.where( 조건 [ , x , y ] ) 
#조건 에 따라 x 또는 y 에서 선택한 요소를 반환 합니다 .
pitch_index = np.where((left_f>20.0) & (left_f<4200.0))# 20.0 < 4200.0 사이의 
print("pitch_index",pitch_index)
pitch_freq = left_f[pitch_index]
pitch_mag = left_spectrum[pitch_index]

mag_max = np.max(pitch_mag)
max_index = np.where(pitch_mag == mag_max)
print(max_index)
print(pitch_freq[max_index])

def convertFregToPitch(arr):
    return 39.86*np.log10(arr/440.0) + 69.0
print(convertFregToPitch(pitch_freq[max_index]))

# 도(C4)~시(B4) 까지의 주파수 범위로 좁혀서, 최대값의 주파수를 확인