# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from lpsd import lpsd


def main():
    N = int(1e5)  # Number of data points in the timeseries
    fs = 2.  # sampling rate
    fmin = float(fs) / N  # lowest frequency of interest
    fmax = float(fs) / 2.  # highest frequency of interest
    Jdes = 1000  # desired number of points in the spectrum
    Kdes = 100  # desired number of averages
    Kmin = 2  # minimum number of averages
    xi = 0.5  # fractional overlap
    x = np.random.normal(size=N)
    X, f, C = lpsd(x, np.hanning, fmin, fmax, Jdes, Kdes, Kmin, fs, xi)

    # Compare to Pwelch
    nfft = np.ceil(fs / float(fmin))
    # Pxx, f_Pwelch = pwelch(x, hanning(nfft), 0, nfft, fs)  # TODO translate to Python
    # loglog(f_Pwelch, Pxx, 'color', [0 0.5 0])  # TODO translate to Python
    # loglog(f, X .* C.PSD, 'color', [0 0.8 0], 'linewidth', 5)  # TODO translate to Python

    plt.figure()
    plt.plot(f, X)
    plt.show()


if __name__ == '__main__':
    main()
