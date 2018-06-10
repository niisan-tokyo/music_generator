# -*- coding: utf-8 -*-
import sys

import glob
import params as par
import numpy as np

l1_testfiles = glob.glob(par.l1_encoded_dir + '*')
#l1_testfiles = l1_testfiles[:6]

temps = []
for filename in l1_testfiles:
    temps.append(np.load(filename))

l1_train_data = np.concatenate(temps)
print(l1_train_data.shape)
np.random.shuffle(l1_train_data)
np.save(par.l2_train_filename, l1_train_data)
