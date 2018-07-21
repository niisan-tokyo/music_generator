# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import params as par
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D

train_files = par.encodedfile_for_test

x_train = np.load(train_files['x_train'])
y_train = np.load(train_files['y_train'])
x_test = np.load(train_files['x_test'])
y_test = np.load(train_files['y_test'])

if Path(par.generator_model).exists():
    model = load_model(par.generator_model)
else:
    model = Sequential()
    model.add(Conv1D(64, 6, activation='relu', input_shape=(2400, 16), padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 6, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 6, activation='relu', padding='same'))
    model.add(Conv1D(16, 6, padding='same'))

model.summary()
model.compile(loss='mse', optimizer='adam')

if (len(sys.argv) > 0):
    epochs = int(sys.argv[1])
else:
    epochs = 100

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

model.save(par.generator_model)