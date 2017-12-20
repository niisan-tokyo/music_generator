# -*- coding: utf-8 -*-
import sys
sys.path.append('/notebooks')

import glob
import numpy as np
import os.path

input_files = glob.glob('/data/input/*encoded.npy')

in_data = []
output = []
for filename in input_files:
    data = np.load(filename)
    seq = len(data)
    for i in range(seq - 45):
        in_data.append(data[i:i+45])

in_data = np.array(in_data)
np.random.shuffle(in_data)

np.save('/data/input/composed_random_input', in_data)
