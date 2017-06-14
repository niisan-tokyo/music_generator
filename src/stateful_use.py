# -*- coding: utf-8 -*-
import wave
import struct
from scipy import fromstring, int16
import numpy as np
#from pylab import *
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
#%matplotlib inline

wavfile = '/data/input/battle1.wav'
wr = wave.open(wavfile, "rb")
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()

model = load_model('/data/model/mcreator')

N = 256
step = 2 ** 8 #512
steps = 128
batch = 32
start = 416500
frames = 200
samples = 32

origin = wr.readframes(wr.getnframes())
data = origin[start:start+N * samples * (steps + 1) * 4]
wr.close()

X = np.frombuffer(data, dtype="int16") /  32768.0
left = X[::2]
right = X[1::2]

# 関数溜まり

def fourier (x, n, w):
    K = []
    for i in range(0, w):
        sample = x[i * n:( i + 1) * n]
        partial = np.fft.fft(sample) / (n / 2)
        K.append(partial)

    return K

def inverse_fourier (k):
    ret = []
    for sample in k:
        inv = np.fft.ifft(sample) * N / 2
        ret.extend(inv.real)

    print (len(sample))
    return ret

def create_test_data(left, right):
    arr = []
    for i in range(0, len(right)):
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

def data_spliter(data, dim):
    Kl_dash = data[:, :dim] + data[:, dim:2 * dim] * 1j
    Kr_dash = data[ :, 2 * dim:3*dim] + data[:, 3 * dim:4*dim] * 1j

    return Kl_dash, Kr_dash

def combine_wav (left, right,N):
    ret = []
    number = len(right) if len(left) > len(right) else len(left)
    for i in range(0, number - 1):
        data = [left[i], right[i]]
        ret.extend(data)

    return np.array(ret)

Kl = fourier(left, N, samples * steps)
Kr = fourier(right, N, samples * steps)
sample = create_test_data(Kl, Kr)
sample = np.reshape(sample, (samples, steps, 4 * N))
music = []

for i in range(steps):
    in_data = sample[:, i, :]
    model.predict(np.reshape(in_data, (samples, 1, 4 * N)))
    #music.append(np.reshape(in_data, (32, 4 * N)))

for i in range(0, frames):
    if i % 50 == 0:
        print('progress: ', i, '/', frames)

    music_data = model.predict(np.reshape(in_data, (samples, 1, 4 * N)))
    music.append(np.reshape(music_data, (samples, 4 * N)))
    in_data = music_data

music = np.array(music)

print(music.shape)
music = np.reshape(music, (frames * samples, 4*N))
print(music.shape)

kl, kr = data_spliter(music, N)
print(kl.shape)

raw_data = combine_wav(inverse_fourier(kl), inverse_fourier(kr), N)
raw_data = raw_data[N * step:] *  32768
#plot(raw_data[1:2000])
raw_data = raw_data.astype('int16')
outf = '/data/output/test.wav'
outd = struct.pack("h" * len(raw_data), *raw_data)
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(width)
ww.setframerate(fr)
ww.writeframes(outd)
ww.close()
