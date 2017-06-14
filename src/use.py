import wave
import struct
from scipy import fromstring, int16
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

# parameter set

wavfile = '/data/input/battle2.wav'
wr = wave.open(wavfile, "rb")
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()

model = load_model('/data/model/mcreator2')

N = 256
step = 2 ** 8 #512
steps = 512
batch = 32
start = 416500
frames = 2000

origin = wr.readframes(wr.getnframes())
data = origin[start:start+N * (steps + 1) * 4]
wr.close()

X = np.frombuffer(data, dtype="int16") /  32768.0
left = X[::2]
right = X[1::2]

# function definition
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

#======================

# use model to create music
Kl = fourier(left, N, step)
Kr = fourier(right, N, step)
sample = create_test_data(Kl, Kr)
music = []

for in_data in sample:
    music.append(np.reshape(in_data, (1, 4 * N)))

for i in range(0, frames):
    if i % 100 == 0:
        print('progress: ', i, '/', frames)

    music_data = model.predict(np.reshape(sample, (1, step, 4 * N)))
    music.append(music_data)
    sample = np.delete(sample, 0, 0)
    sample = np.append(sample, music_data, 0)

music = np.array(music)
#===============

# coodinate music data
print(music.shape)
music = np.reshape(music, (frames + step, 4*N))
print(music.shape)

kl, kr = data_spliter(music, N)
print(kl.shape)
#===============

# output wave file
raw_data = combine_wav(inverse_fourier(kl), inverse_fourier(kr), N)
raw_data = raw_data[N * step:] *  32768
raw_data = raw_data.astype('int16')
outf = '/data/output/test.wav'
outd = struct.pack("h" * len(raw_data), *raw_data)
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(width)
ww.setframerate(fr)
ww.writeframes(outd)
ww.close()

#===============
