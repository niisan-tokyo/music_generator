# -*- coding: utf-8 -*-
import wave
import struct
import glob
import os.path
import numpy as np
from mylibs import constants as con

test_files = glob.glob('/data/input/*.wav')
test_files = test_files[:2]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:con.fr * 4 * 270]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    X = np.reshape(X, (-1, con.fr // 2))
    #print(X.shape)
    #print(len(X))
    return X

arr = []
for file in test_files:
    arr.append(get_dataset(file))

raw_data = np.array(arr)
raw_data = np.reshape(raw_data, (-1, con.fr // 4, 2))
np.random.shuffle(raw_data)

np.save('/data/input/raw_wave.npy', raw_data)
