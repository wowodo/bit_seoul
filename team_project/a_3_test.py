import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import timeit

file_path = 'audio/nsynth-test/audio_path/bass_electronic_018-022-100.wav'

start_time = timeit.default_timer()# 시작 시간 체크

y, sr = librosa.load(file_path)
# y : 파의 진동의 폭

#sr : sampling_rate (샘플링 주파수) sr = 22050대상 샘플링 속도
#이산적 신호를 만들기 연속적 신호에서 초당  샘플링 횟수를 정의 하는 단위 Hz

#FFt(Fast fourier Transform) 고속 푸리에 
#시간영역의 신호를 주파수 영역으로 변환해주는 역할을 한다
fft = np.fft.fft(y)/len(y)

magnitude = np.abs(fft) #절대 값으로 취해서  magnitude 구하기
#푸리에는 다시 음원으로 역변이 가능 하지만  abs 해주면 역변이 안됨
#음수의 수를 양수로 올려줌

#mp.linspace (시작, 끝, num=50,  끝점)

f = np.linspace(0,sr,len(magnitude))#몇개의 일정한 간격으로 요소를 만들기

#푸리에를 통과한 스펙트럼은 대칭구조로 나와서  high frequency(높은 진동수)
#  부분 절반을 날리고 앞쪽 절반만 사용
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

pitch_index = np.where((left_f>20.0) & (left_f<4200.0))
print(pitch_index) #(array([   80,    81,    82, ..., 16797, 16798, 16799], dtype=int64),)

pitch_freq = left_f[pitch_index]#x축 
pitch_mag = left_spectrum[pitch_index]#y축left_spectrum[pitch_index]

def convertfregToPitch(arr):
    #np.round(반올림)해서 수 많은 소주점들을 하나로 합쳐줌
    return np.round(39.86*np.log10(arr/440.0) +  69.0)
convertfregToPitch2 = np.vectorize(convertfregToPitch)
pitch_freq = convertfregToPitch2(pitch_freq)

start_index = np.where(pitch_freq>=21)
print("start_index : ", start_index)#start_index :  (array([   27,    28,    29, ..., 16717, 16718, 16719], dtype=int64),)
pitch_freq = pitch_freq[start_index]
pitch_mag = pitch_mag[start_index]
print(pitch_freq)
print(pitch_mag)
print("pitch_freq.shape", pitch_freq.shape)
print("pitch_mag.shape", pitch_mag.shape)

freq_uniq = np.unique(pitch_freq)
print(freq_uniq[0])

temp_arr = []
for cnt in range(freq_uniq.shape[0]):
    temp_avg = np.average(pitch_mag[np.where(pitch_freq==freq_uniq[cnt])])
    temp_arr = np.insert(temp_arr, cnt, temp_avg)

print(temp_arr.shape)
plt.xlabel("MIDI number")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()
terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

# 주파수별로 중복된 값을 합쳐서
# 미디번호 21~108 사이 그래프가 되도록 정리함

# file_path = './data/same_pitch_diff_inst/reed_acoustic_037-060-127.wav'

print(int(file_path[-11:-8]))
# 60