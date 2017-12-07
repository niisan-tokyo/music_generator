# -*- coding: utf-8 -*-
import wave
import struct
import glob
from mylibs import constants as con
from scipy import fromstring, int16
import numpy as np
import os.path
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Reshape
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam

frame = con.fr // 4

if os.path.exists(con.model_encoder):
    model = load_model(con.model_encoder)
else:
    model = Sequential()
    model.add(Conv1D(64, 8, padding='same', input_shape=(frame, 2), activation='relu'))
    model.add(MaxPooling1D(5, padding='same'))
    model.add(Conv1D(64, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(5, padding='same'))
    model.add(Conv1D(64, 4, padding='same', activation='relu'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(Conv1D(32, 4, padding='same', activation='relu'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(Conv1D(16, 4, padding='same', activation='relu'))
    model.add(UpSampling1D(3))
    model.add(Conv1D(32, 4, padding='same', activation='relu'))
    model.add(UpSampling1D(3))
    model.add(Conv1D(64, 4, padding='same', activation='relu'))
    model.add(Conv1D(50, 8, padding='same', activation='tanh'))
    model.add(Reshape((frame, 2)))

model.compile(loss='mse', optimizer='adam')

raw_data = np.load('/data/input/raw_wave.npy')
np.random.shuffle(raw_data)

# for epoch in range(5):
#     for filename in test_files:
#         data = get_dataset(filename)
#         print('start')
#         model.fit(data, data, validation_split=0.1, epochs=5)
#
#     print('epoch ', epoch, 'end')
model.fit(raw_data, raw_data, validation_split=0.05, epochs=20)

model.save(con.model_encoder)
