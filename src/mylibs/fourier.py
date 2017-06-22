import numpy as np
from mylibs import constants as con

def hamming_fourier (x):
    K = []
    n = con.N
    HAMMING = np.hamming(con.N)
    factor = con.fourier_factor
    for i in range(0, 2 * con.span - 1):
        start = i * n // 2
        sample = x[start:start + n] * HAMMING
        partial = np.fft.fft(sample) * factor
        K.append(partial[: n // 2])

    return K

def inverse_hamming_fourier(k):
    ret = []
    n = con.N
    HAMMING = np.hamming(n)
    factor = con.inverse_fourier_factor
    for sample in k:
        inv = np.fft.ifft(sample) / HAMMING
        ret.extend(inv.real / factor)

    print (len(sample))
    return ret

def fourier(x):
    K = []
    n = con.N
    factor = con.fourier_factor
    for i in range(0, con.span - 1):
        sample = x[i * n:(i + 1) * n]
        partial = np.fft.fft(sample) * factor
        K.append(partial[:n // 2])

    return K

def inverse_fourier(k):
    ret = []
    factor = con.inverse_fourier_factor
    for sample in k:
        inv = np.fft.ifft(sample)
        ret.extend(inv.real / factor)

    print (len(ret))
    return ret
