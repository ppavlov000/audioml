import numpy as np


def fft_init(hop_size, fft_size):
    win = np.zeros(fft_size)
    win[hop_size:] = np.hanning(fft_size - hop_size)
    return win

def fft_analysis(in_buf, tmp_buff, win, fft_size):
    hop_size = len(in_buf)
    ibfr = np.zeros(fft_size)
    tmp_buff = np.concatenate([tmp_buff, in_buf])
    ibfr = tmp_buff[hop_size:]
    X = win * ibfr
    X = np.fft.rfft(X)
    return X, ibfr


def fft_synthesis(in_buf, tmp_buff, hop_size):
    fft_size = 2*(len(in_buf) - 1)
    ibfr = np.zeros(fft_size)
    y = np.fft.irfft(in_buf, n=fft_size)
    tmp_buff = tmp_buff + y
    tmp_buff = np.concatenate([tmp_buff, np.zeros(hop_size)])
    ibfr = tmp_buff[hop_size:fft_size + hop_size]
    return tmp_buff[:hop_size], ibfr



