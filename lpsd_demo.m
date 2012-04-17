%% Try out lpsd

N = 1e5;        % number of data points in the timeseries
fs = 2;         % sampling rate

fmin = fs/N;    % lowest frequency of interest
fmax = fs/2;    % highest frequency of interest
Jdes = 1000;    % desired number of points in the spectrum

Kdes = 100;     % desired number of averages
Kmin = 2;       % minimum number of averages

xi = 0;         % fractional overlap

x = randn(N, 1);

[X, f, C] = lpsd(x, @hanning, fmin, fmax, Jdes, Kdes, Kmin, fs, xi);

%% Compare to Pwelch
nfft = ceil(fs/fmin);
[Pxx, f_Pwelch] = pwelch(x, hanning(nfft), 0, nfft, fs);
loglog(f_Pwelch, Pxx, 'color', [0 0.5 0]);
hold all
loglog(f, X .* C.PSD, 'color', [0 0.8 0], 'linewidth', 5);
hold off