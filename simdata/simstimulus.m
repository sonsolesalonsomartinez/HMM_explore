function [stim,isi] = simstimulus(T,Q,mean_ISI,var_ISI,range_ISI,stim_probabilities)
% Simulate train of stimuli, for sessions of length given by T. Separation
% between presentation of the stimulus is random, as well as which stimulus
% is presented each time
% T: length of each session
% Q: number of classes of stimulus
% mean_ISI: mean inter-stimulus interval (ISI). 
% var_ISI: variance of the ISIs. 
% range_ISI: maximum and minimum possible value for ISIs
% stim_probabilities: probability of each stimulus

if nargin < 6, stim_probabilities = ones(1,Q)/Q; end
if nargin < 5, range_ISI = [350 5000]; end
if nargin < 4, var_ISI = 100; end
if nargin < 3, mean_ISI = 750; end

N = length(T); 
sigma = var_ISI / mean_ISI;

stim_cumprobabilities = cumsum(stim_probabilities); 

stim = zeros(sum(T),1);
isi = [];

for j = 1:N
    s = zeros(T(j),1);
    t = 1; 
    while 1
        p = rand(1);
        c = find(stim_cumprobabilities>=p,1);
        isi = [isi round(sigma*poissrnd(mean_ISI/sigma))];
        if isi(end) < range_ISI(1), isi(end) = range_ISI(1); end
        if isi(end) > range_ISI(2), isi(end) = range_ISI(2); end
        t = t + isi(end);
        if t > (T(j)-100), break; end
        s(t) = c;
    end
    t = (1:T(j)) + sum(T(1:j-1));
    stim(t) = s;
end

end



