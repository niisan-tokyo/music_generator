# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import wave
import struct
import glob
import os.path
import importlib
import numpy as np

name = sys.argv[1] + '_model'
if (os.path.exists(name + '.py') == False):
    raise Exception('そんなファイルは存在しない')

level = importlib.import_module(name)
level.model.compile(loss='mse', optimizer='adam')
level.model.summary()

raw_data = np.load(level.train_filename)
print(raw_data.shape)

level.model.fit(raw_data, raw_data, validation_split=0.05, epochs=level.epochs)

level.model.save(level.encoder_file)
