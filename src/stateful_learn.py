# -*- coding: utf-8 -*-
import wave
import struct
import glob
from scipy import fromstring, int16
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def get_dataset(filename, samples, span, offset=0):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[offset:samples * span * 4 + offset]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    return X[::2], X[1::2]

# samples
N = 1024
fr = 44100
offset = fr * 2

# learning params
samples = 32 #2 ^5 +1
batch_num = samples * 128
span = batch_num + 100
dims = 4 * N
epochs = 30
test_files = glob.glob('/data/input/*.wav')
print(test_files)
test_files = test_files[1:2]
files_num = len(test_files)


def fourier (x, n, w):
    K = []
    for i in range(0, w-2):
        sample = x[i * n:( i + 1) * n]
        partial = np.fft.fft(sample) * 100
        K.append(partial)

    return K

def inverse_fourier (k):
    ret = []
    for sample in k:
        inv = np.fft.ifft(sample)
        ret.extend(inv.real)

    print (len(sample))
    return ret

def create_test_data(left, right):
    arr = []
    for i in range(0, len(right)-1):
        #複素数のベクトル化
        temp = np.array([])
        temp = np.append(temp, left[i].real)
        temp = np.append(temp, left[i].imag)
        temp = np.append(temp, right[i].real)
        temp = np.append(temp, right[i].imag)
        arr.append(temp)

    return np.array(arr)

def pack_step(data, batch_num, samples, dims):
    temp = data[:batch_num + samples]
    test = temp.reshape((batch_num + samples, 1, dims))
    return test
    # arr = []
    # for i in range(batch_num // samples + 1):
    #     for n in range(samples):
    #         arr.append(temp[i * samples +n])
    # return np.reshape(np.array(arr), (batch_num + samples, 1, dims))


test = []
for filename in test_files:
    left, right = get_dataset(filename, N, span, offset=offset)
    Kl, Kr = fourier(left, N, span), fourier(right, N, span)
    data = create_test_data(Kl, Kr)
    data_rs = pack_step(data, batch_num, samples, dims)
    test.append(data_rs)

#test = np.reshape(test, (files_num * samples, steps + 1, dims))

model = Sequential()
model.add(LSTM(256,
              input_shape=(1, dims),
              batch_size=samples,
              #output_shape=(None, dims),
              return_sequences=True,
              #activation='tanh',
              stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(256, stateful=True, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, stateful=True, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256, stateful=True, return_sequences=False))
model.add(Dense(256, activation='relu'))
model.add(Dense(dims))
model.compile(loss='mse', optimizer='rmsprop')

for num in range(0, epochs):
    print(num + 1, '/', epochs, ' start')
    for one_data in test:
        in_data = one_data[:-samples]
        out_data = np.reshape(one_data[samples:], (batch_num, dims))
        model.fit(in_data, out_data, epochs=1, shuffle=False, batch_size=samples)

        model.reset_states()
    print(num+1, '/', epochs, ' epoch is done!')

model.save('/data/model/mcreator12')
