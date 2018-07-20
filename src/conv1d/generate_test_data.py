# -*- coding: utf-8 -*-
import sys

import glob
import params as par
import numpy as np

l1_testfiles = glob.glob(par.l1_encoded_dir + '*')
#l1_testfiles = l1_testfiles[:6]

# temps = []
# for filename in l1_testfiles:
#     temps.append(np.load(filename))

def create_dataset(data):
    x_train = []
    y_train = []
    maximum = len(data) - par.input_second - par.output_second
    for i in range(maximum):
        x_train.append(np.reshape(data[i:i+par.input_second], (-1, 16)))
        y_train.append(np.reshape(data[i+par.input_second:i+par.input_second + par.output_second], (-1, 16)))

    return (np.array(x_train), np.array(y_train))

def load_data(filenames):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for filename in filenames:
        dataset = create_dataset(np.load(filename))
        print(dataset[0].shape)
        x_train.append(dataset[0][:420])
        y_train.append(dataset[1][:420])
        x_test.append(dataset[0][420:])
        y_test.append(dataset[1][420:])

    return (np.concatenate(x_train), np.concatenate(y_train), np.concatenate(x_test), np.concatenate(y_test))

loadeddata = load_data(l1_testfiles)
print(loadeddata[0].shape)
print(loadeddata[1].shape)
print(loadeddata[2].shape)
print(loadeddata[3].shape)

# l1_train_data = np.concatenate(temps)
# print(l1_train_data.shape)
# np.random.shuffle(l1_train_data)
# np.save(par.l2_train_filename, l1_train_data)
