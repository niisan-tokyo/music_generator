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
from keras.layers import Dense, Flatten, Reshape, Activation
from keras.engine.topology import Input
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv1D, UpSampling1D, Conv2DTranspose
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.advanced_activations import PReLU


if os.path.exists(par.l3_encoder_filename):
    model = load_model(par.l3_encoder_filename)
else:
    model = Sequential()
    model.add(Conv1D(32, 21, strides=21, activation='relu', use_bias=False, padding='same', input_shape=(par.l1_input_length, par.l1_channel_size)))
    model.add(Conv1D(16, 7, strides=7, activation='relu', use_bias=False, padding='same'))
    model.add(Reshape((1, 150, 16)))
    #model.add(Conv2DTranspose(32, (1, 2), use_bias=False, strides=(1, 2), activation='relu'))
    model.add(Conv2DTranspose(32, (1, 3), use_bias=False, strides=(1, 3), activation='relu'))
    model.add(Conv2DTranspose(32, (1, 7), use_bias=False, strides=(1, 7), activation='relu'))
    model.add(Conv2DTranspose(16, (1, 7), use_bias=False, strides=(1, 7), activation='relu'))
    #model.add(UpSampling1D(7))
    #model.add(Conv1D(32, 14, use_bias=False, padding='same', activation='relu'))
    #model.add(UpSampling1D(7))
    #model.add(Conv1D(32, 14, use_bias=False, padding='same', activation='relu'))
    #model.add(UpSampling1D(6))
    #model.add(Conv1D(32, 16, use_bias=False, padding='same', activation='relu'))
    #model.add(Conv1D(par.l1_channel_size, 16, use_bias=False, padding='same'))
    model.add(Reshape((22050, 16)))
    K.set_floatx = 'float16'
    model.add(Conv1D(2, 14, use_bias=False, padding='same'))
    

train_filename = par.l1_train_filename
epochs = par.l1_epochs
encoder_file = par.l3_encoder_filename

encoder = K.function([model.layers[0].input], [model.layers[1].output])
decoder = K.function([model.layers[2].input], [model.layers[7].output])
