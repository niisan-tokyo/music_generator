# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')
import re
import wave
import struct
import glob
import params as par
from scipy import fromstring, int16
import numpy as np
import os.path
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import Callback
from keras import backend as K

test_files = glob.glob(par.base_data + 'input/*.wav')

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:par.fr * 4 * 240]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    X = np.reshape(X, (-1, par.l1_input_length, par.l1_channel_size))
    return X

import l3_model as level1

print(test_files)

for filename in test_files:
    data = get_dataset(filename)
    output = level1.encoder([data])
    savename = re.sub('.*\/', '', filename).replace('.wav', '.npy')
    savename = re.sub('.*\\\\', '', savename)
    print(savename)
    print(output[0].shape)
    np.save(par.l1_encoded_dir + savename, output[0])
