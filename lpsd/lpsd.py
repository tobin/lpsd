import numpy as np
from scipy.signal import get_window


def lpsd(x, fs, window, fmin, fmax, Jdes=1000, Kdes=100, Kmin=1, xi=0.5):
    """
    Compute the LPSD power spectrum estimation with a logarithmic frequency axis.

    Parameters
    ----------
    x : array_like
        time series to be transformed. "We assume to have a long stream x(n),
        n=0, ..., N-1 of equally spaced input data sampled with frequency fs. Typical
        values for N range from 10^4 to >10^6" [1]

    fs : float
        Sampling frequency of the `x` time series.

    window : str
        Desired window to use. If `window` is a string or tuple, it is passed to
        `scipy.signal.get_window` to generate the window values, which are DFT-even by
        default. See `scipy.signal.get_window` for a list of windows and required
        parameters. "Choose a window function w(j, l) to reduce spectral leakage within
        the estimate. ... The computations of the window function will be performed when
        the segment lengths L(j) have been determined." [1]

    fmin, fmax : float, optional
        Lowest and highest frequency to estimate. Defaults to `fs / len(x)` and the
        Nyquist frequency `fs / 2`, respectively. "... we propose not to use the first
        few frequency bins. The first frequency bin that yields unbiased spectral
        estimates depends on the window function used. The bin is given by the effective
        half-width of the window transfer function." [1].

    Jdes : int, optional
        Desired number of Fourier frequencies. Defaults to 1000. "A typical value for J
        is 1000" [1]

    Kdes : int, optional
        Desired number of averages. Defaults to 100.

    Kmin : int, optional
        Minimum number of averages. Defaults to 1.

    xi : float, optional
        Fractional overlap between segments (0 <= xi < 1). Defaults to 0.5. "The
         amount of overlap is a trade-off between computational effort and flatness of
        the data weighting." [1]. See Figures 5 and 6 [1].

    Returns
    -------
    Pxx : 1d-array
        Vector of (uncalibrated) power spectrum estimates
    f : 1-d array
        Vector of frequencies corresponding to Pxx
    C : dict
        Dict containing calibration factors to calibrate Pxx into either power spectral
        density or power spectrum.

    Notes
    -----
    The implementation follows references [1] and [2] quite closely; in particular, the
    variable names used in the program generally correspond to the variables in the
    paper; and the corresponding equation numbers are indicated in the comments.

    References
    ----------
      [1] Michael Tröbs and Gerhard Heinzel, "Improved spectrum estimation  from
      digitized time series on a logarithmic frequency axis" in Measurement, vol 39
      (2006), pp 120-129.
        * http://dx.doi.org/10.1016/j.measurement.2005.10.010
        * http://pubman.mpdl.mpg.de/pubman/item/escidoc:150688:1

      [2] Michael Tröbs and Gerhard Heinzel, Corrigendum to "Improved spectrum
      estimation from digitized time series on a logarithmic frequency axis."

    """

    N = len(x)  # Table 1
    jj = np.arange(Jdes, dtype=int)  # Table 1

    if not fmin:
        fmin = fs / N  # Lowest frequency possible
    if not fmax:
        fmax <= fs / 2  # Nyquist rate

    g = np.log(fmax) - np.log(fmin)  # (12)
    f = fmin * np.exp(jj * g / (Jdes - 1))  # (13)
    rp = fmin * np.exp(jj * g / (Jdes - 1)) * (np.exp(g / (Jdes - 1)) - 1)  # (15)

    # r' now contains the 'desired resolutions' for each frequency bin, given the rule
    # that we want the resolution to be equal to the difference in frequency between
    # adjacent bins. Below we adjust this to account for the minimum and desired number
    # of averages.

    ravg = (fs / N) * (1 + (1 - xi) * (Kdes - 1))  # (16)
    rmin = (fs / N) * (1 + (1 - xi) * (Kmin - 1))  # (17)

    case1 = rp >= ravg  # (18)
    case2 = np.logical_and(rp < ravg, np.sqrt(ravg * rp) > rmin)  # (18)
    case3 = np.logical_not(np.logical_or(case1, case2))  # (18)

    rpp = np.zeros(Jdes)

    rpp[case1] = rp[case1]  # (18)
    rpp[case2] = np.sqrt(ravg * rp[case2])  # (18)
    rpp[case3] = rmin  # (18)

    # r'' contains adjusted frequency resolutions, accounting for the finite length of
    # the data, the constraint of the minimum number of averages, and the desired number
    # of averages.  We now round r'' to the nearest bin of the DFT to get our final
    # resolutions r.
    L = np.around(fs / rpp).astype(int)  # segment lengths (19)
    r = fs / L  # actual resolution (20)
    m = f / r  # Fourier Tranform bin number (7)

    # Allocate space for some results
    Pxx = np.empty(Jdes)
    S1 = np.empty(Jdes)
    S2 = np.empty(Jdes)

    # Loop over frequencies. For each frequency, we basically conduct Welch's method
    # with the fourier transform length chosen differently for each frequency.
    # TODO: Try to eliminate the for loop completely, since it is unpythonic and slow.
    # Maybe write doctests first...
    for jj in range(len(f)):

        # Calculate the number of segments
        D = int(np.around((1 - xi) * L[jj]))  # (2)
        K = int(np.floor((N - L[jj]) / D + 1))  # (3)

        # reshape the time series so each column is one segment  <-- FIXME: This is not
        # clear.
        a = np.arange(L[jj])
        b = D * np.arange(K)
        ii = a[:, np.newaxis] + b  # Selection matrix
        data = x[ii]  # x(l+kD(j)) in (5)

        # Remove the mean of each segment.
        data -= np.mean(data, axis=0)  # (4) & (5)

        # Compute the discrete Fourier transform
        w = get_window(window, L[jj])  # (5)
        sinusoid = np.exp(
            -2j * np.pi * np.arange(L[jj])[:, np.newaxis] * m[jj] / L[jj]
        )  # (6)
        data = data * (sinusoid * w[:, np.newaxis])  # (5,6)

        # Average the squared magnitudes
        Pxx[jj] = np.mean(np.abs(np.sum(data, axis=0)) ** 2)  # (8)

        # Calculate some properties of the window function which will be used during
        # calibration
        S1[jj] = np.sum(w)  # (23)
        S2[jj] = np.sum(w ** 2)  # (24)

    # Calculate the calibration factors
    C = {"PS": 2.0 / (S1 ** 2), "PSD": 2.0 / (fs * S2)}  # (28)  # (29)

    return Pxx, f, C
