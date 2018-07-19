# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import wave
import re
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
print(test_files)
#test_files = test_files[5]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:par.fr * 4 * 60]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ (par.bit_depth * 1.0)
    X = np.reshape(X, (-1, par.l1_input_length, par.l1_channel_size))
    #print(X.shape)
    #print(len(X))
    return X

import l3_model as level3

for test_file in test_files:
    data = get_dataset(test_file)
    temp = level3.encoder([data])
    res = level3.decoder([temp[0]])
    row_data = np.array(res)
    mdata = np.reshape(res, (-1)) * par.bit_depth
    mdata = mdata.astype('int16')
    savename = re.sub('.*\/', '', test_file)
    savename = re.sub('.*\\\\', '', savename)
    outf = par.base_data + 'output/' + savename
    outd = struct.pack("h" * len(mdata), *mdata)
    ww = wave.open(outf, 'w')
    ww.setnchannels(2)
    ww.setsampwidth(2)
    ww.setframerate(par.fr)
    ww.writeframes(outd)
    ww.close()
