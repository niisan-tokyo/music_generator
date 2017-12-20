# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import wave
import struct
import glob
from mylibs import constants as con
from scipy import fromstring, int16
import numpy as np
import os.path
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import Callback
from keras import backend as K

test_files = glob.glob('/data/input/*.wav')
test_files = test_files[:2]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:con.fr * 4 * 270]
    wr.close()
    X = np.frombuffer(data, dtype="int16") / 32768.0
    return X

if os.path.exists(con.model_encoder):
    model = load_model(con.model_encoder)
else:
    model = Sequential()
    model.add(Dense(512, activation='sigmoid', input_dim=con.fr // 2))
    model.add(Dense(con.fr // 2))

model.compile(loss='mse', optimizer='adam')

raw_data = np.load('/data/input/raw_wave.npy')
#arr = []
#for file in test_files:
#    arr.append(get_dataset(file))

#raw_data = np.array(arr)
#raw_data = np.reshape(raw_data, (-1, con.fr // 2))
#np.random.shuffle(raw_data)

# for epoch in range(5):
#     for filename in test_files:
#         data = get_dataset(filename)
#         print('start')
#         model.fit(data, data, validation_split=0.1, epochs=5)
#
#     print('epoch ', epoch, 'end')
model.fit(raw_data, raw_data, validation_split=0.05, epochs=con.epochs)

model.save(con.model_encoder)
