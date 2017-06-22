# -*- coding: utf-8 -*-

# サンプリング範囲
N = 1024
EN = N // 2 + 1
fr = 44100
offset = fr * 2

# 学習用パラメータ
samples = 32 #2 ^5 +1
batch_num = samples * 160
span = batch_num + 100
dims = 4 * EN
neuron = 512
epochs = 10

# フーリエのよくわからない因子
fourier_factor = 10
inverse_fourier_factor = 10

# モデル名
model_name = '/data/model/mcreator16'
