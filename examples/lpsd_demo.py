# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from lpsd import lpsd
from scipy.signal import 


def main():
    N = int(1e5)  # Number of data points in the timeseries
    fs = 2.0  # sampling rate
    fmin = float(fs) / N  # lowest frequency of interest
    fmax = float(fs) / 2.0  # highest frequency of interest
    Jdes = 1000  # desired number of points in the spectrum
    Kdes = 100  # desired number of averages
    Kmin = 2  # minimum number of averages
    xi = 0.5  # fractional overlap
    x = np.random.normal(size=N)
    X, f, C = lpsd(x, np.hanning, fmin, fmax, Jdes, Kdes, Kmin, fs, xi)

    # Compare to Pwelch
    nfft = np.ceil(fs / fmin)
    window = np.hanning(nfft)
    f_Pwelch, Pxx = signal.welch(x, fs, window, nfft=nfft, scaling="density")

    plt.figure()
    plt.loglog(f_Pwelch, Pxx)
    plt.loglog(f, X * C["PSD"])
    plt.show()


if __name__ == "__main__":
    main()
