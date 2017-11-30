# -*- coding: utf-8 -*-
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

if os.path.exists(con.model_encoder):
    model = load_model(con.model_encoder)
else:
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=con.fr // 2))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(con.fr // 16, activation='relu'))
    #model.add(Dense(512, activation='sigmoid'))
    #model.add(Dense(con.fr // 16, activation='relu'))
    #model.add(Dense(con.fr // 8, activation='relu'))
    model.add(Dense(con.fr // 2, activation='tanh'))

model.compile(loss='mse', optimizer='adam')

raw_data = np.load('/data/input/raw_wave.npy')

# for epoch in range(5):
#     for filename in test_files:
#         data = get_dataset(filename)
#         print('start')
#         model.fit(data, data, validation_split=0.1, epochs=5)
#
#     print('epoch ', epoch, 'end')
model.fit(raw_data, raw_data, validation_split=0.05, epochs=15)

model.save(con.model_encoder)
