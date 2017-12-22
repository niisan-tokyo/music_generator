# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import glob
from mylibs import constants as con
from scipy import fromstring, int16
import numpy as np
import os.path
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D

if os.path.exists(con.model_composer):
    model = load_model(con.model_composer)
else:
    model = Sequential()
    model.add(LSTM(256, input_shape=(40, 512), return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=True))
    model.add(Reshape((5, 512)))

model.compile(loss='mse', optimizer='adam')

raw_data = np.load('/data/input/composed_random_input.npy')
in_data = raw_data[:, :40, :]
out_data = raw_data[:, 40:, :]

print(model.summary())
print(in_data.shape)
print(out_data.shape)

model.fit(in_data, out_data, validation_split=0.05, epochs=4)

model.save(con.model_composer)
