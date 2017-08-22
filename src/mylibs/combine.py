from mylibs import constants as con, fourier
import numpy as np

def data_spliter(data):
    num = con.EN
    Kl_dash = data[:, : num] + data[:, num:2 * num] * 1j
    Kl_dash = np.append(Kl_dash, np.flip(np.conj(Kl_dash[:, 1:-1]), 1), 1)
    Kr_dash = data[:, 2 * num:3 * num] + data[:, 3 * num:4 * num] * 1j
    Kr_dash = np.append(Kr_dash, np.flip(np.conj(Kr_dash[:, 1:-1]), 1), 1)
    print(Kl_dash.shape)

    return Kl_dash, Kr_dash

def combine_wav (left, right):
    ret = []
    number = len(right) if len(left) > len(right) else len(left)
    for i in range(0, number - 1):
        data = [left[i], right[i]]
        ret.extend(data)

    return np.array(ret)

def normal_combine(kl, kr):
    left = fourier.inverse_fourier(kl)
    right = fourier.inverse_fourier(kr)

    return combine_wav(left, right)

def hamming_combine (kl, kr):
    left = fourier.inverse_hamming_fourier(kl)
    right = fourier.inverse_hamming_fourier(kr)
    left = hamming_summation(np.array(left))
    right = hamming_summation(np.array(right))

    return combine_wav(left, right)

def hamming_summation(k):
    ret = np.zeros(len(k))
    size = len(k) // con.N
    for i in range(size):
        n = i * con.N
        ret[n // 2:n // 2 + con.N] += k[n:n+con.N]

    return ret
