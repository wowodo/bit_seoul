import numpy as np
import matplotlib.pyplot as plt
import librosa

file_path = 'audio/nsynth-test/audio_path/bass_electronic_018-022-100.wav'

y, sr = librosa.load(file_path)
# y : 파의 진동 폭을 말한다

#sr : sampling rate 또는 sampling frequency (샘플링 주파수)를 뜻한다  sr = 22505

#fft(Fast fourier Transform) 즉 고속 푸리에 변환이 되겠습니다.
#계속 변화하는 데이터를 주파수 영역으로 가져다가 어떤 주파수 들이 사용되는 확인할때 사용
fft = np.fft.fft(y)/len(y) 

magnitude = np.abs(fft) 
#보통 푸리에 변환은 역변환이 가능지만 절대 값을 거쳐 스펙트로그램을 만들었기 때문에 역변환이 불가능

#np.linsapce (시작, 정지,  num=50) 지정된 간격으로 균등하게 숫자를 반환합니다. 
#간격을 통해 계산된 num[시작, 중지]를 반환합니다.간격의 끝점을 선택적으로 제외할 수 있습니다
f = np.linspace (0, sr, len(magnitude))


#푸리에 변환을 통과한  specturm은 대칭 구조로 나와서 high frequency 부분 절반을 날리고 앞쪽 절반 사용
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

print(type(left_f))#<class 'numpy.ndarray'>
print(left_f.shape)#(44100,)
print(left_spectrum.shape)#(44100,)

mag_max = np.max(left_spectrum) #left_spectrum #최대 값을 저장시킨다

print("mag_max: ",mag_max) #mag_max:  0.0008241585436276515

# 
max_index = np.where(left_spectrum == mag_max)  #값을 비교해서

print(left_f[max_index])# [130.50147961]

def convertPitch(arr):
    if arr != 0:
        return 39.86*np.log10(arr/440.0) + 69.0
convertPitch2 = np.vectorize(convertPitch)
left_f = convertPitch2(left_f)


print(left_f[max_index]) #[47.960407581303194]

plt.scatter(left_f, left_spectrum)
plt.xlabel("MIDI number")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()

# 주파수를 미디번호로 바꾸는 공식 찾음
# 또한 array에 커스텀함수를 돌리려면 vectorize하면 됨