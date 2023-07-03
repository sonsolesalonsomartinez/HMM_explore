function cstim = convolveStimulus(stim,modulation,delay,jitter)
% Convolves the stimulus spikes with a continuous (made-up) kernel
% type='phase':  asymmetric Gaussian kernel with quick ramping up and sharp decay
% type='power','signal_inj','frequency', symmetric log kernels
% the response has a mean delay (3nd argument) and delay variance (4rd argument)
if strcmp(modulation,'phase')
    x = linspace(-4,0,50); y = normpdf(x); % gaussian kernel
elseif strcmp(modulation,'frequency')
    x = linspace(-2,2,400); y = -log(abs(x).^10+1); y = y-min(y);% log kernel
elseif strcmp(modulation,'power')
    %x = linspace(-4,4,1001); y = normpdf(x); % gaussian kernel
    %y = y(unique([1:10:501 501:2:1001])); y = y(1:end-1);
    x = linspace(-2,2,400); y = -log(abs(x).^10+1); y = y-min(y);% log kernel
elseif strcmp(modulation,'signal_inj')
    x = linspace(-2,2,200); y = -log(abs(x).^10+1); y = y-min(y);% log kernel
else
    error('Wrong type of modulation')
end
y = y / max(y); L = length(y); y = y';
cstim = zeros(length(stim),1);
tstim = find(stim)'; 
for t = tstim
    d = round(delay + jitter*randn(1)); if d<0, d=0; end
    yd = [zeros(d,1); y]; 
    if (t+L-1+d) > length(stim)
        tmax = length(cstim(t:end));
        cstim(t:end) = yd(1:tmax) + cstim(t:end);
    else
        cstim(t:t+L-1+d) = yd + cstim(t:t+L-1+d);
    end
end
cstim(cstim>1) = 1; 
end