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

test_files = glob.glob('/home/ec2-user/music_generator/data/input/*.wav')
test_files = test_files[:1]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:par.fr * 4 * 180]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0

    term = par.l1_input_length // 3
    num = 0
    Y = []
    while term * (num + 3) < len(X):
        Y.append(X[term * num: term * (num + 3)])
        num = num + 1
    Y = np.reshape(np.array(Y), (-1, par.l1_input_length))

    return Y

arr = []
for file in test_files:
    arr.append(get_dataset(file))

processed_arr = []


raw_data = np.array(arr)

# 畳み込みにかけるため、基本データ集合を(N, 1)のshapeに変更
raw_data = np.reshape(raw_data, (-1, par.l1_input_length, 1))
np.random.shuffle(raw_data)

np.save(par.l1_train_filename, raw_data)
