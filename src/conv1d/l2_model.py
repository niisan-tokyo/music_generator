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
from keras.layers import Dense, Flatten, Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras import backend as K

if os.path.exists(par.l2_encoder_filename):
    model = load_model(par.l2_encoder_filename)
else:
    model = Sequential()
    model.add(Conv1D(par.l2_first_filters, 14, padding='same', input_shape=par.l2_input_shape, activation='relu'))
    model.add(MaxPooling1D(par.l2_first_strides))
    model.add(Conv1D(par.l2_second_filters, 14, padding='same', activation='relu'))
    model.add(MaxPooling1D(par.l2_second_strides))
    model.add(Conv1D(par.l2_third_filters, 14, padding='same', activation='relu'))
    model.add(Conv1D(par.l2_final_filters, 14, padding='same', activation='relu'))
    model.add(Conv1D(par.l2_third_filters, 14, padding='same', activation='relu'))
    model.add(UpSampling1D(par.l2_second_strides))
    model.add(Conv1D(par.l2_second_filters, 14, padding='same', activation='relu'))
    model.add(UpSampling1D(par.l2_first_strides))
    model.add(Conv1D(par.l2_first_filters, 14, padding='same', activation='relu'))
    model.add(Conv1D(par.l1_second_filters, 16, padding='same'))

train_filename = par.l2_train_filename
epochs = par.l2_epochs
encoder_file = par.l2_encoder_filename

encoder = K.function([model.layers[0].input], [model.layers[5].output])
decoder = K.function([model.layers[5].input], [model.layers[11].output])
