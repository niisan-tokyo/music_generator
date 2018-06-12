# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import wave
import struct
import glob
import os.path
import importlib
import numpy as np
from keras.callbacks import ModelCheckpoint

name = sys.argv[1] + '_model'
if (os.path.exists(name + '.py') == False):
    raise Exception('そんなファイルは存在しない')

level = importlib.import_module(name)
level.model.compile(loss='mse', optimizer='adam')
level.model.summary()

f = open('model.json', 'w')
f.write(level.model.to_json())
f.close()

if (len(sys.argv) > 2):
    epochs = int(sys.argv[2])
else:
    epochs = level.epochs

raw_data = np.load(level.train_filename)
print(raw_data.shape)

'''
バリデーションロスが最小の時に保存
ある程度学習が進んでいるときや、モデルが不安定な時に使う
'''
#checkpointer = ModelCheckpoint(filepath=level.encoder_file, save_best_only=True)
#level.model.fit(raw_data, raw_data, validation_split=0.05, epochs=epochs, callbacks=[checkpointer])

'''
規定回数の学習終了後に保存
'''
level.model.fit(raw_data, raw_data, validation_split=0.05, epochs=epochs)
level.model.save(level.encoder_file)
