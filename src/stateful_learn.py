# -*- coding: utf-8 -*-
import wave
import struct
import glob
from mylibs import constants as con
from scipy import fromstring, int16
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


test_files = glob.glob('/data/input/*.data.npy')

model = Sequential()
model.add(LSTM(con.neuron,
              input_shape=(1, con.dims),
              batch_size=con.samples,
              #output_shape=(None, dims),
              return_sequences=True,
              #activation='tanh',
              stateful=True))
model.add(Dropout(0.5))
model.add(LSTM(con.neuron, stateful=True, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(con.neuron, stateful=True, return_sequences=False))
model.add(Dense(con.dims))
model.compile(loss='mse', optimizer='adam')

for num in range(0, con.epochs):
    print(num + 1, '/', con.epochs, ' start')
    for filename in test_files:
        one_data = np.load(filename)
        in_data = one_data[:-con.samples]
        out_data = np.reshape(one_data[con.samples:], (con.batch_num, con.dims))
        model.fit(in_data, out_data, epochs=1, shuffle=False, batch_size=con.samples)

        model.reset_states()
    print(num+1, '/', con.epochs, ' epoch is done!')

model.save(con.model_name)
