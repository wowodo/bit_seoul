import numpy as np
import matplotlib.pyplot as plt
import librosa
import timeit


file_path = 'audio/nsynth-test/audio_path/bass_electronic_018-022-100.wav'

y, sr = librosa.load(file_path)

fft = np.fft.fft(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()
