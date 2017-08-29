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


test_files = glob.glob('/data/input/*.data.npy')

def tanh(x):
    return con.fourier_factor * K.tanh(x)

model = Sequential()
model.add(LSTM(con.neuron,
              input_shape=(1, con.dims),
              batch_size=con.samples,
              #output_shape=(None, dims),
              return_sequences=True,
              activation='tanh',
              stateful=True))
model.add(LSTM(con.neuron, stateful=True, return_sequences=True, activation='sigmoid'))
model.add(LSTM(con.neuron, stateful=True, return_sequences=False, activation='selu'))
model.add(Dense(con.dims))
model.compile(loss='mse', optimizer='adam')

if (os.path.exists(con.model_name)):
    model = load_model(con.model_name)

test_files = test_files[1:2]

class ResetStates(Callback):
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, batch, logs={}):
        self.model.reset_states()

for num in range(0, con.epochs):
    print(num + 1, '/', con.epochs, ' start')
    for filename in test_files:
        one_data = np.load(filename)
        in_data = one_data[:-con.samples]
        out_data = np.reshape(one_data[con.samples:], (con.batch_num, con.dims))
        model.fit(in_data, out_data, callbacks=[ResetStates()], epochs=con.local_epocks, shuffle=False, batch_size=con.samples)

    print(num+1, '/', con.epochs, ' epoch is done!')

model.save(con.model_name)
