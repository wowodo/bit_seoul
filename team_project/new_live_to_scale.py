import pyaudio
import numpy as numpy
import matplotlib.pyplot as plt
import pickle
import datetime
CHUNK = 2250
RATE = 2250 #음성 데이터의 샘플링 레이트이다.
SR = 2250
OFFSET = 48

def transe(y):
    tmp = (int(y) - OFFSET) % 12
    result = ''
    if tmp == 0:
        result = '도'
    elif tmp == 2:
        result = '레'
    elif tmp == 4:
        result = '미'
    elif tmp == 5:
        result = '파'
    elif tmp == 6:
        result = '솔'
    elif tmp == 8:
        result = '라'
    elif tmp == 10:
        result = '시'
    return result

        ##전처리 과정이 똑같다 왜냐하면 midi 파일로 같이 바꿔줘야 하기 때문에
def preprocessing(data):
    fft = np.fft.fft(data)
    magnitude = np.abs(fft)
    f = np.linspace(0,SR,len(magnitude)) 
    left_spectrum = magnitude[:int(len(magnitude)/2)]
    left_f = f[:int(len(magnitude)/2)]
    pitch_index = np.where((left_f > 130.0) & (left_f < 1050.0)) #130 ~ 1050 헤르츠의 index 구함
    pitch_freq = left_f[pitch_index] #x축 
    pitch_mag = left_spectrum[pitch_index] #y축


    def convertFregToPitch(arr):
            return np.round(39.86*np.log10(arr/440.0) + 69.0) #수 많은 소수점 들을 하나로 합치게 해줌. Ex 130.8 130.9 130.10 을 전부 130 => 48로 단일화 즉 값들이 48로 몰링
    convertFregToPitch2 = np.vectorize(convertFregToPitch)

    pitch_freq = convertFregToPitch2(pitch_freq)

    start_index = np.where(pitch_freq>=48)
    pitch_freq = pitch_freq[start_index]
    pitch_mag = pitch_mag[start_index]
    freq_uniq = np.unique(pitch_freq)

        tmp_arr = []
    for i in range(len(freq_uniq)):
        tmp_avg = np.average(pitch_mag[np.where(pitch_freq == freq_uniq[i])]) # 48을 가진 index 들을 모두 가져와서 avg
        tmp_arr.append(tmp_avg)

# 백색 소음의 파워를 찾아 준것 50 이하로 소리는 무시하는 것
    if max(pitch_mag) < 50:    
        result = np.array(tmp_arr)
        return result, False
        # 50이 넘어 가서 True가 나올때는 
    else:
        result = np.array(tmp_arr)
        result = result.reshape(1,result.shape[0])
        return result, True

################################################################################################
p=pyaudio.PyAudio()
#스트림  pyaudio 를 통해 소리를 얻어 온다
stream=p.open(format=pyaudio.paFloat32,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)
 #CHUNK 어느 단위 만큼 얻어 온다 
 #np.frombuffer 바이너리 데이터를 숫자로 변환해준다
while(True):
    start_time = datetime.datetime.now()
    data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
    pre_data, is_sound  = preprocessing(data) #숫자로 변환해서 넘기면 전처리가된  x_predict 를 반환한다
    # print(data)
    if is_sound: #모델을 프레딕트 하고 transe로 와서 반환한것 48(도)
        model = pickle.load(open('./model/modelLoad/modelFolder/lgbm.dat', 'rb'))
        y_predict = model.predict(pre_data)
        result = transe(y_predict) # 48을 도로 바꿔준다 
        print(result)
        print()
    else:
        print('소리 없음')
    # print(pre_data.shape)
    # plt.plot([i for i in range(48,85)],pre_data)
    # plt.show()
    print("한 번 출력까지 걸린 시간 : ",datetime.datetime.now() - start_time)

stream.stop_stream()
stream.close()
p.terminate()
