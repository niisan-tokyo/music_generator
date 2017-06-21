# -*- coding: utf-8 -*-
import wave
import struct
import glob
from scipy import fromstring, int16
import numpy as np
from mylibs import fourier, constants as con
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
    test = temp.reshape((batch_num + samples, 1, dims))
    return test

test_files = glob.glob('/data/input/*.wav')
for filename in test_files:
    left, right = get_dataset(filename, con.N, con.span, offset=con.offset)
    Kl, Kr = fourier.hamming_fourier(left), fourier.hamming_fourier(right)
    data = create_test_data(Kl, Kr)
    data_rs = pack_step(data, con.batch_num, con.samples, con.dims)
    outfile = filename.replace('.wav', '.data')
    np.save(outfile, data_rs)
