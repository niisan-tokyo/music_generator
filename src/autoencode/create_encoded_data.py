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
from keras import backend as K

test_files = glob.glob('/data/input/*.wav')
test_files = test_files[:7]

model = load_model(con.model_encoder)
inter_model = K.function([model.layers[0].input], [model.layers[0].output])

def get_dataset(filename):
    wavfile = filename
    wr = wave.open(wavfile, "rb")
    origin = wr.readframes(wr.getnframes())
    data = origin[:con.fr * 4 * 180]
    wr.close()
    X = np.frombuffer(data, dtype="int16")/ 32768.0
    X = np.reshape(X, (-1, con.fr // 2))
    return X

def output_encoded(filename, wave_data, inter_model):
    # for frame_data in wave_data:
    #     layer = inter_model([np.reshape(frame_data, (1, con.fr // 2))])
    #     ret.append(layer)
    output = inter_model([wave_data])
    output = np.reshape(np.array(output), (-1, 512))
    print(output.shape)

    np.save(filename.replace('.wav', '') + '_encoded', output)

for filename in test_files:
    print(filename)
    output_encoded(filename, get_dataset(filename), inter_model)
