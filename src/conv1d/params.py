# -*- coding: utf-8 -*-
# param for audio
fr = 44100    # frame rate
frst = fr * 2 # stereo

import os

base_src = os.getenv('BASE_SRC', '/notebooks/')
base_data = os.getenv('BASE_DATA', '/data/')

# param for level1
l1_encoder_filename = base_data + 'model/convl1encoded'
l1_train_filename = base_data + 'input/l1_encoded_wav.npy'
l1_encoded_dir = base_data + 'input/conv_l1/'
l1_input_length = fr // 2

l1_first_strides = 7
l1_first_filters = 8

l1_second_strides = 7
l1_second_filters = 16

l1_epochs = 20

# param for level2
l2_encoder_filename = base_data + 'model/convl2encoded'
l2_train_filename = base_data + 'input/l2_encoded_wav.npy'
l2_encoded_dir = base_data + 'input/conv_l2'
l2_input_shape = (l1_input_length // (l1_first_strides * l1_second_strides), l1_second_filters)

l2_first_strides = 3
l2_first_filters = 32

l2_second_strides = 2
l2_second_filters = 26
l2_third_filters = 20
l2_final_filters = 16

#l2_final_filters = 32

l2_epochs = 100
