import numpy as np
import matplotlib.pyplot as plt
import librosa

file_path = 'audio/nsynth-test/audio_path/bass_electronic_025-061-075.wav'


y, sr = librosa.load(file_path)
# y : 파의 진동의 폭

#sr : sampling rate (샘플링 주파수)를 뜻한다  sr = 22050 대상 샘플링 속도
#이산적인 신호를 만들기위해 연속적 신호에서 얻어진 단위 초당 샘플링 횟수를 정의 단위는 Hz


#fft(Fast fourier Transform)즉 고속 푸리에 변환
#시간 영역의 신호를 주파수 영역으로 변환해주는 역활을 한다
fft = np.fft.fft(y) 

magnitude = np.abs(fft)
#절대값을 취해서 magnitude 구하기
#보통 푸리에 변환은 음원으로 변형도 가능하지만 np.abs(fft)하면 역변 안됨

#np.linspace (시작, 정지, num=50, 끝점=True, retstep=거짓, dtype=없음, 축=0)
f = np.linspace(0, sr, len(magnitude)) #몇개의 일정한 간격으로 요소를 만들기 /규모 크기

#푸리에 변환을 통과한 spectrum 은 대칭구조로 나와서  high frequency부분 절반을 날리고
#앞쪽 절반만 사용한다
left_spectrum = magnitude[int:(len(magnitude)/2)] #절반 잘라서 대칭안되게 하기
left_f = f[int:(len(magnitude)/2)]

plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")# x쪽 라벨
plt.ylabel("magnitude")# y쪽 라벨
plt.title("power spectrum") #맨 위쪽 타이틀
plt.show()

pitch_index = np.where((left_f > 260.0) &(left_f < 494.0 )) # 260 ~ 494헤르츠의 index
print("pitch_index", pitch_index)
pitch_freq = left_f[pitch_index]
pitch_mag = left_spectrum[pitch_index]

mag_max = np.max(pitch_mag)
max_index = np.where(pitch_mag == mag_max)
print(max_index)
print(pitch_freq[max_index])

def convertFregToPitch(arr):
    return 39.86*np.log10(arr/440.0) + 69.0
print(convertFregToPitch(pitch_freq[max_index]))

#도(c4)~시(b4) 까지의 주파수 범위로 좁혀서 최대값의 주파수를 확인




ch_freq = left_f[pitch_index] # x축
ch_mag = left_spectrum[pitch_index] #y축
