# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

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

test_files = glob.glob('/data/input/*.wav')
test_files = test_files[0]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:par.fr * 4 * 30]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    X = np.reshape(X, (-1, par.l1_input_length, 1))
    #print(X.shape)
    #print(len(X))
    return X

import l1_model as level1
import l2_model as level2

data = get_dataset(test_files)
res = []
#for datum in data:
#    res.append(model.predict(np.reshape(datum, (1, par.l1_input_length, 1))))

temp = level1.encoder([data])
temp = level2.encoder([temp[0]])
temp =level2.decoder([temp[0]])
res = level1.decoder([temp[0]])

row_data = np.array(res)
mdata = np.reshape(res, (-1)) * 32768
mdata = mdata.astype('int16')
outf = '/data/output/test.wav'
outd = struct.pack("h" * len(mdata), *mdata)
ww = wave.open(outf, 'w')
ww.setnchannels(2)
ww.setsampwidth(2)
ww.setframerate(par.fr)
ww.writeframes(outd)
ww.close()
