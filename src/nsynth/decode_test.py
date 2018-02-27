import os
import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

filename = '/data/input/battle1_encoded.npy'
sr = 44100

encoding = np.load(filename)

fastgen.synthesize(encoding, save_paths=['/data/output/test.wav'], samples_per_save=11050, checkpoint_path="/data/model/wavenet-ckpt/model.ckpt-200000")
