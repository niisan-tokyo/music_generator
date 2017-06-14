# -*- coding: utf-8 -*-
import wave
import struct
import glob
from scipy import fromstring, int16
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

def get_dataset(file, samples, span):
    wavfile = '/data/input/' + file + '.wav'
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:samples * span * 4]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    return X[::2], X[1::2]

# samples
N = 256
span = 18000
fr = 44100

# learning params
step   = 2 ** 8
steps = 512 # 2 ^ 9
batch_size = 32
samples = batch_size
dims = 4 * N
epochs = 1
test_files = glob.glob('/data/input/*.wav')
files_num = len(test_files)

# functions
def fourier (x, n, w):
    K = []
    for i in range(0, w-2):
        sample = x[i * n:( i + 1) * n]
        partial = np.fft.fft(sample) / (n / 2)
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

def pack_step(data, samples, steps, dims):
    temp = data[:samples * steps]
    test = temp.reshape((samples, steps, dims))
    return test

# テスト用のデータ生成
test = []
for file in test_files:
    print('o-ke')
    left, right = get_dataset(file, N, span)
    Kl, Kr = fourier(left, N, span), fourier(right, N, span)
    data = create_test_data(Kl, Kr)
    data_rs = pack_step(data, samples, steps, dims)
    test.append(data_rs)

model = Sequential()
print((None, step, dims))
model.add(LSTM(256,
              input_shape=(step, dims),
              #output_shape=(None, dims),
              return_sequences=False,
              activation='relu'))
#model.add(LSTM(256, return_sequences=True))
#model.add(LSTM(256, return_sequences=False))
model.add(Dense(dims, activation='tanh'))
model.compile(loss='mse', optimizer='rmsprop')

for epoch in range(epochs):
    for in_data in test:
        for i in range(0, steps - step -1):
            loss = model.train_on_batch(in_data[:, i:i+step, :], in_data[:, i+step, :])
            if (i % 100 == 50):
                print('i: ', i, ', loss: ', loss)

    print(epoch, '/', epochs, ' is done')

model.save('/data/model/mcreator2')
