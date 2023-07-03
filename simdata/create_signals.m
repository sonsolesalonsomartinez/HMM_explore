%% create a signal of specified amplitude and frequency

function signal=create_signals(Fc, A, StopTime, phi, Fs, noise)
% inputs: 
% FC = frequency of the signal in hertz - mandatory!
% A = amplitude of the signal - mandatory!
% StopTime = time in sec of the signal to produce - mandatory!
% phi = phase of the signal (in degrees) - default = 0
% Fs = sampling frequency - default = 500 Hz
% plot_signal = boolean - if to visualize the signal or not - default = 0

% output:
% signal = a sinewave with specified Fc, A, phi, of StopTime*(Fs) length
if nargin<=3
    Fs = 500;  % sampling frequency (resolution of the signal)
    phi = 0;
    noise = 0;
end


dt = 1/Fs;                   % timestep size
t = (0:dt:StopTime-dt)';     % time array 
%%Sine wave:
x = A * cos(2*pi*Fc*t + phi * 2 * pi);          % signal itself

signal = x +  noise * randn(size(x));
% Plot the signal versus time:

end


