# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import wave
import struct
import glob
from mylibs import constants as con
import os.path
from scipy import fromstring, int16
import numpy as np

test_files = glob.glob('/data/input/*.wav')
test_files = test_files[:10:4]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:con.fr * 4 * 180]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0

    term = con.fr // 6
    num = 0
    Y = []
    while term * (num + 3) < len(X):
        Y.append(X[term * num: term * (num + 3)])
        num = num + 1
    Y = np.reshape(np.array(Y), (-1, con.fr // 2))

    return Y

arr = []
for file in test_files:
    arr.append(get_dataset(file))

processed_arr = []


raw_data = np.array(arr)
raw_data = np.reshape(raw_data, (-1, con.fr // 2))
np.random.shuffle(raw_data)

np.save('/data/input/raw_wave.npy', raw_data)
