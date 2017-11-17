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
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D

test_files = glob.glob('/data/input/*.wav')
test_files = test_files[:2]

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:con.fr * 4 * 180]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    X = np.reshape(X, (-1, con.fr // 4, 2))
    #print(X.shape)
    #print(len(X))
    return X

#for filename in test_files:
#    get_dataset(filename)

if os.path.exists(con.model_encoder):
    model = load_model(con.model_encoder)
else:
    model = Sequential()
    model.add(Conv1D(16, 8, padding='same', input_shape=(con.fr // 4 , 2), activation='relu'))
    model.add(MaxPooling1D(5, padding='same'))
    model.add(Conv1D(8, 4, padding='same', activation='relu'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(Conv1D(8, 4, padding='same', activation='relu'))
    model.add(MaxPooling1D(3, padding='same'))
    model.add(Conv1D(4, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(4, 3, padding='same', activation='relu'))
    model.add(UpSampling1D(5))
    model.add(Conv1D(8, 4, padding='same', activation='relu'))
    model.add(UpSampling1D(3))
    model.add(Conv1D(8, 4, padding='same', activation='relu'))
    model.add(UpSampling1D(3))
    model.add(Conv1D(16, 8, padding='same', activation='relu'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(1, 8, padding='same', activation='tanh'))

model.compile(loss='mae', optimizer='adam')

arr = []
for file in test_files:
    arr.append(get_dataset(file))

raw_data = np.array(arr)
raw_data = np.reshape(raw_data, (-1, con.fr // 4, 2))
np.random.shuffle(raw_data)

# for epoch in range(5):
#     for filename in test_files:
#         data = get_dataset(filename)
#         print('start')
#         model.fit(data, data, validation_split=0.1, epochs=5)
#
#     print('epoch ', epoch, 'end')
model.fit(raw_data, raw_data, validation_split=0.05, epochs=10)

model.save(con.model_encoder)
