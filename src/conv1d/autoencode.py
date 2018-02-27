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
from keras.layers import Dense, Flatten, Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras import backend as K

if os.path.exists(con.model_encoder):
    model = load_model(con.model_encoder)
else:
    model = Sequential()
    model.add(Conv1D(8, 7, strides=7, padding='same', input_shape=(con.fr // 2, 1), activation='relu'))
    model.add(Conv1D(16, 7, strides=7, padding='same', input_shape=(con.fr // 2, 1), activation='relu'))
    model.add(Conv1D(32, 5, strides=5, padding='same', input_shape=(con.fr // 2, 1), activation='relu'))
    model.add(Conv1D(16, 5, strides=3, padding='same', input_shape=(con.fr // 2, 1), activation='relu'))
    model.add(UpSampling1D(7))
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(7))
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(5))
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(3))
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(Conv1D(1, 8, padding='same', activation='tanh'))


model.compile(loss='mse', optimizer='adam')
model.summary()

raw_data = np.load('/data/input/raw_wave.npy')
print(raw_data.shape)
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
