# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import wave
import struct
import glob
import params as par
import os.path
from scipy import fromstring, int16
import numpy as np

test_files = glob.glob(par.base_data + 'input/*.wav')
test_files = test_files[:2]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:par.fr * 4 * 180]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0

    term = par.l1_input_length * par.l1_channel_size // 4
    num = 0
    Y = []
    while term * (num + 4) < len(X):
        Y.append(X[term * num: term * (num + 4)])
        num = num + 1
    Y = np.reshape(np.array(Y), (-1, par.l1_input_length, par.l1_channel_size))

    return Y

arr = []
for file in test_files:
    arr.append(get_dataset(file))

processed_arr = []


raw_data = np.array(arr)

# 畳み込みにかけるため、基本データ集合を(N, 1)のshapeに変更
raw_data = np.reshape(raw_data, (-1, par.l1_input_length, par.l1_channel_size))
np.random.shuffle(raw_data)

np.save(par.l1_train_filename, raw_data)
