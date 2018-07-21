# -*- coding: utf-8 -*-
# param for audio
fr = 44100    # frame rate
frst = fr * 2 # stereo

bit_depth = 2**10

import os

base_src = os.getenv('BASE_SRC', '/Users/niiku/Documents/project/music_generator/src/')
base_data = os.getenv('BASE_DATA', '/Users/niiku/Documents/project/music_generator/data/')

# param for level1
l1_encoder_filename = base_data + 'model/convl1encoded_onceup'
l1_train_filename = base_data + 'input/l3_encoded_wav.npy'
l1_encoded_dir = base_data + 'input/conv_l1/'
l1_input_length = fr // 2
l1_channel_size = 2

l1_first_strides = 21
l1_first_filters = 32

l1_second_strides = 14
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

l3_encoder_filename = base_data + 'model/convl3encoded9'

input_second = 8 * 2
output_second = 2 * 2

encodedfile_for_test = {
    'x_train': base_data + 'input/encoded_waves.npy',
    'y_train': base_data + 'input/encoded_waves_label.npy',
    'x_test': base_data + 'input/encoded_waves_test.npy',
    'y_test': base_data + 'input/encoded_waves_test_label.npy'
}

generator_model = base_data + 'model/generator'