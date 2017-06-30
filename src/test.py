# -*- coding: utf-8 -*-
import wave
import struct
from scipy import fromstring, int16
import numpy as np
from mylibs import fourier

wavfile = '/data/input/battle1.wav'
wr = wave.open(wavfile, "rb")
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()

N = 1024
span = 3500

print('cn', ch)
print('fr', fn)
print('sm', 1.0 * N * span / fr, 'sec')

origin = wr.readframes(wr.getnframes())
data = origin[:N * span * ch * width]
wr.close()

print('len', len(origin))
print('smlen: ', len(data))

X = np.frombuffer(data, dtype="int16")
left = X[::2]
right = X[1::2]

def myfourier (x, n, w):
    K = []
    for i in range(0, w-2):
        sample = x[i * n:( i + 1) * n]
        partial = np.fft.fft(sample)
        K.append(partial)

    return K

def myinverse_fourier (k):
    ret = []
    for sample in k:
        inv = np.fft.ifft(sample)
        ret.extend(inv.real)

    print (len(sample))
    return ret

Kl = fourier.fourier(left)
Kr = fourier.fourier(right)

def combine_wav (left, right):
    ret = []
    number = len(right) if len(left) > len(right) else len(left)
    for i in range(0, number -1):
        data = [int(left[i]), int(right[i])]
        ret.extend(data)

    return ret

left_dash = fourier.inverse_fourier(Kl)
right_dash = fourier.inverse_fourier(Kr)
raw_data = combine_wav(left_dash, right_dash)

outd = struct.pack("h" * len(raw_data), *raw_data)

outf = '/data/output/test.wav'
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(width)
ww.setframerate(fr)
ww.writeframes(outd)
ww.close()
