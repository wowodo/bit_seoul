import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import os
FiG_SIZE =(8,6) #피규어 사이즈 상수로 고정(반복문에 들어 갈 필요 없으니까)

# aa = "keyboard_electronic_069-067-050"
# print(aa.split('-'))
# y_label = int(aa.split('-')[-2])
# print(y_label)


#이름에서 48~84이내있는 것들만 
def search(dirname):
    x_arr = []
    y_arr = []
    i = 0
                #os.listdir :모든 디렉터리 목록을 가져오는데 사용한다
    filenames = os.listdir(dirname)
    for filename in filenames:
        y_label = int(filename.split('-')[-2]) 
        #keyboard_electronic_069-067-050 에서 -2 int로 067 -> 67 이 되서 
        if y_label < 48 or 84 < y_label:
            print(y_label)
            continue 
        full_filename = os.path.join(dirname, filename)
        full_filename = full_filename.replace('\\','/')

        sig, sr = librosa.load(full_filename, sr=22050)


        # print(sig,sig.shape)

        #fft(Fast fourier transform)
        #시간영역의 신호를 주파수 영역으로 변환해주는 역할을 한다
        fft = np.fft.fft(sig)
        
        #절대 값으로 magnitude 구하기
        #음수를 양수로 / 음원으로 역변 불가
        magnitude = np.abs(fft)

        #np.linspace(시작 , 끝, num=50)
        #몊개의 일정한 간격으로 요소를 만들기
        f = np.linspace(0, sr,len(magnitude))


        #푸리에 변환을 통과한  spectrum 은 대칭구조가 나와서 high frequency부분 절반을 잘라 내고 사용
        left_spectrum = magnitude[:int(len(magnitude)/2)]
        left_f = f[:int(len(magnitude)/2)]

        
        #여기까지 전처리 1번 
        # =========================================================================
        
        #기존의 hz 는 너무 크다 줄인다
        #130~1050.0  까지
        pitch_index = np.where((left_f > 130.0)&(left_f < 1050.0))
        
        pitch_freq = left_f[pitch_index] #x축
        pitch_mag = left_spectrum[pitch_index] #y 축
        

        def convertFregToPitch(arr):
            return np.round(39.86*np.log10(arr/440.0) + 69.0) #소수점들을 반올림해서 하나로 합친다

        convertFregToPitch2 = np.vectorize(convertFregToPitch)

        pitch_freq = convertFregToPitch2(pitch_freq)
        
        start_index = np.where(pitch_freq >= 48)

        pitch_freq = pitch_freq[start_index]

        #여러 미디번호들이 있지만 유니크로 보여주며 유니크 이전엔
        #유니크로 보여주며 유니크 이전엔 48 48 48 48 48 48 48 이런식으로 있을 것이고 해당 
        # 인덱스로 주면 mag를 얻는다.
        #mag들의 합을 얻으면 가장 큰 mag를 얻을 수 있다.
         # print(freq_uniq[0]) #대략 미디번호 48 ~ 84까지 있을 거다  # 130 hz ~ 1050 hz 까지 잘랐으니
        freq_uniq = np.unique(pitch_freq)

        tmp_arr = []
        for i in range(len(freq_uniq)):
            tmp_avg = np.average(pitch_mag[np.where(pitch_freq == freq_uniq[i])])
            tmp_arr.append(tmp_avg)

        x_arr.append(tmp_arr)
        y_arr.append(y_label)

    return np.array(x_arr), np.array(y_arr)

x,y = search('audio/nsynth-test/nsynth-valid/audio')


print(np.unique(y))
print(len(np.unique(y)))

# np.save('./npy/all_scale_x.npy', arr=x)
# np.save('./npy/all_scale_y.npy', arr=y)
        



