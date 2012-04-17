% Power spectrum estimation with a logarithmic frequency axis
%
% Implementation of the algorithm described by Tröbs and Heinzel in
% _Measurement_, vol 39 (2006), pp 120-129, available here:
% 
% * http://dx.doi.org/10.1016/j.measurement.2005.10.010
% * http://pubman.mpdl.mpg.de/pubman/item/escidoc:150688:1/component/escidoc:150687/sdarticle.pdf
%
% Tobin Fricke
% tobin.fricke@ligo.org
% 2012-04-17

%function X = lpsd(x, windowfcn, fmin, fmax, Jdes, Kdes)

N = 1e5;
fs = 2;
fmin = fs/N;
fmax = fs/2;
Jdes = 1000;
Kdes = 100;
Kmin = 2;       % minimum number of averages
xi = 0;         % Overlap  FIXME: NOT IMPLEMENTED
windowfcn = @hanning;

x = randn(N, 1);

N = length(x);  % Table 1


% Jdes:  desired number of Fourier frequencies
% Kdes:  desired number of averages

jj = 0:Jdes-1;

g = log(fmax) - log(fmin);                                          % (12)

f = fmin * exp((jj * g) / (Jdes - 1));                              % (13)

rp = fmin * exp(jj * g / (Jdes -1)) * (exp(g / (Jdes - 1)) - 1);    % (15)

%%
ravg = (fs/N) * (1 + (1 - xi) * (Kdes - 1));                        % (16)
rmin = (fs/N) * (1 + (1 - xi) * (Kmin - 1));                        % (17)

case1 = rp >= ravg;                                                 % (18)
case2 = (rp < ravg) & (sqrt(ravg * rp) > rmin);                     % (18)
case3 = ~(case1 | case2);                                           % (18)

rpp = zeros(1, Jdes);

rpp( case1 ) = rp(case1);                                           % (18)
rpp( case2 ) = sqrt(ravg * rp(case2));                              % (18)
rpp( case3 ) = rmin;                                                % (18)

L = round(fs ./ rpp);       % segment lengths                       % (19)
r = fs ./ L;                % actual resolution                     % (20)

m = f ./ r;                 % Fourier Tranform bin number           % (7) 

loglog(f, r, '.-', f, L, '.-', f, N./L, '.-')
line(get(gca,'xlim'), Kmin*[1 1], 'color', 'r');
line(get(gca,'xlim'), Kdes*[1 1], 'color', 'r');
legend('resolution', 'segment length', 'number of averages', 'minimum averages', 'desired averages');

%%


X = NaN(1, Jdes);
S1= NaN(1, Jdes);
S2= NaN(1, Jdes);

% Nonzero segment overlap is not yet implemented
assert(xi==0);  

for jj=1:length(f)
  % Split the data stream into segments of length L(jj)
  K = floor(N / L(jj));  % number of averages
  
  % select all the segments we can without zero-padding
  data = x(1:(K*L(jj)));
  
  % reshape the time series so each column is one segment
  data = reshape(data, L(jj), K);
  
  % Remove the mean of each segment.
  data = data - repmat(mean(data), L(jj), 1);                       % (4)
  
  % Multiply each segment with the window function
  window = windowfcn(L(jj));                                        % (5)
  data = data .* repmat(window, 1, K);
  
  % Compute the discrete Fourier transforms
  sinusoid = exp(-2i*pi * (0:L(jj)-1)' * m(jj)/L(jj));              % (6)
  data = data .* repmat(sinusoid, 1, K);
  
  % Average the squared magnitudes
  X(jj) = mean(abs(sum(data)).^2);                                  % (8)
  
  % Calculate some properties of the window function which will be used
  % during calibration  
  S1(jj) = sum(window);                                             % (23)
  S2(jj) = sum(window.^2);                                          % (24)

end

% Calculate the calibration factors

C_PS = 2 * S1.^(-2);                                                % (28)
C_PSD = 2 ./ (fs * S2);                                             % (29)

%% Compare to Pwelch
nfft = ceil(fs/r(1));
[Pxx, f_Pwelch] = pwelch(x, hanning(nfft), 0, nfft, fs);
loglog(f_Pwelch, Pxx, 'color', [0 0.5 0]);
hold all
loglog(f, X .* C_PSD, 'color', [0 0.8 0], 'linewidth', 5);
hold off

