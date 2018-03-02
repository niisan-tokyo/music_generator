# -*- coding: utf-8 -*-
import params as par

print('=================== level 1====================')
print('initial input frames: %d' % (par.l1_input_length))

print('timestep reduce factor: %d' % (par.l1_first_strides * par.l1_second_strides))

print('filter nums: %d' % (par.l1_second_filters))

timestep = par.l1_input_length / (par.l1_first_strides * par.l1_second_strides)
print('output dim: (%d, %d)' % (timestep, par.l1_second_filters))
print('total params num: %d' % (timestep * par.l1_second_filters))

print('=================== level 2====================')
print('initial input shape: (%d, %d)' % (par.l2_input_shape[0], par.l2_input_shape[1]))

print('timestep reduce factor: %d' % (par.l2_first_strides * par.l2_second_strides))

print('filter nums: %d' % (par.l2_second_filters))

timestep = par.l2_input_shape[0] / (par.l2_first_strides * par.l2_second_strides)
print('output dim: (%d, %d)' % (timestep, par.l2_final_filters))
print('total params num: %d' % (timestep * par.l2_final_filters))
