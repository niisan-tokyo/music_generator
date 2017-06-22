# -*- coding: utf-8 -*-
import wave
import struct
import glob
from scipy import fromstring, int16
import numpy as np
from mylibs import fourier, combine, constants as con
import re

def get_dataset(filename, samples, span, offset=0):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[offset:samples * span * 4 + offset]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    return X[::2], X[1::2]

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
    test = temp.reshape((-1, 1, dims))
    print(test.shape)
    return test

test_files = glob.glob('/data/input/*.wav')
#test_files = test_files[1:2]
for filename in test_files:
    left, right = get_dataset(filename, con.N, con.span, offset=con.offset)
    Kl, Kr = fourier.fourier(left), fourier.fourier(right)
    data = create_test_data(Kl, Kr)
    data_rs = pack_step(data, con.batch_num, con.samples, con.dims)
    outfile = filename.replace('.wav', '.data')
    np.save(outfile, data_rs)

outputtest = '/data/output/test.wav'
filename = test_files[0]
outfile = filename.replace('.wav', '.data.npy')
print(outfile)
data = np.load(outfile)
Kl, Kr = combine.data_spliter(np.reshape(data, (-1, con.dims)))
print(Kl.shape)
print(Kr.shape)
raw = combine.normal_combine(Kl, Kr)
#left = fourier.inverse_fourier(Kl)
#right = fourier.inverse_fourier(Kr)
#raw = combine.combine_wav(left, right)
raw_data = raw[:] *  32768
length = len(raw_data)
print(len(raw_data))
raw_data = raw_data.astype('int16')
outf = '/data/output/test.wav'
outd = struct.pack("h" * len(raw_data), *raw_data)
ww = wave.open(outf, 'w')
ww.setnchannels(2)
ww.setsampwidth(2)
ww.setframerate(con.fr)
ww.writeframes(outd)
ww.close()
