import os
import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

filename = '/data/input/battle1.wav'
sr = 44100
audio = utils.load_audio(filename, sample_length=(sr * 4), sr=sr)
sample_length = audio.shape[0]

print('{} samples, {} seconds'.format(sample_length, sample_length / float(sr)))

encoding = fastgen.encode(audio, '/data/model/wavenet-ckpt/model.ckpt-200000', sample_length)

print (encoding.shape)
np.save(filename.replace('.wav', '') + '_encoded.npy', encoding)

fastgen.synthesize(encoding, save_paths=['/data/output/test.wav'], samples_per_save=sample_length, checkpoint_path="/data/model/wavenet-ckpt/model.ckpt-200000")
