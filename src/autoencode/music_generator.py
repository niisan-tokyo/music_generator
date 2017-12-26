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

generator  = load_model(con.model_encoder)
encoder = K.function([generator.layers[0].input], [generator.layers[0].output])
decoder = K.function([generator.layers[1].input], [generator.layers[1].output])
composer = load_model(con.model_composer_lstm)

'''
first step:

We get wave data from a wav file.
'''
wr = wave.open('/data/input/battle2.wav', "rb")
origin = wr.readframes(wr.getnframes())
data = origin[:con.fr * 4 * 30]
wr.close()
seed = np.frombuffer(data, dtype="int16") / 32768.0
seed = np.reshape(seed, (-1, con.fr // 2))

'''
second step:

The flagment of wave data is processed and we get encoded data.
This encoded data is a music score "created by machine".
'''
encoded = np.array(encoder([seed[:40, :]]))
print(encoded.shape)

'''
third step:

We generate a next music score data by using "composer model".
'''
composed = []
input_data = encoded
for i in range(40):
    output = composer.predict(input_data)
    input_data = np.concatenate((input_data[:, 5:, :], output), axis=1)
    composed.append(output)

score = np.array(composed)
print(score.shape)
score = np.reshape(score, (-1, 512))
print(score.shape)

'''
forth step:

We can perform a music by decoding score to wave data.
The performed data is argumented to /data/output by wav data format.
'''
performance = np.array(decoder([score]))
print(performance.shape)
mdata = np.reshape(performance, (-1)) * 32768.0
mdata = mdata.astype('int16')
outf = '/data/output/test.wav'
outd = struct.pack("h" * len(mdata), *mdata)
ww = wave.open(outf, 'w')
ww.setnchannels(2)
ww.setsampwidth(2)
ww.setframerate(con.fr)
ww.writeframes(outd)
ww.close()
