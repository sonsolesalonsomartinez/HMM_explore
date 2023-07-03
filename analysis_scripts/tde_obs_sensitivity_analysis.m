%% SENSITIVITY ANALYSIS ON TDE OBSERVATION MODEL

%% SCRIPT DESCRIPTION
% This script contains the code for the sensitivity analysis on the TDE 
% observation model, producing results in Figure 2 and Supplementary Figure 1.

%% ANALYSIS 
% To properly simulate the HMM inference, we reasoned the best way is to 
% train multiple models on various training signals and then test which 
% model performs best on a target test signal. 
% this way it is like creating k states (k training models) and 
% check which state best represents the test signal - instead of our
% original approach of training one model on one signal and testing on
% which test signal it performs best

%% AUTHOR
% Laura Masaracchia 
% lauramacfin.au.dk
% Aarhus University
% June 2023

%%
%% TEST TDE frequency resolution varying amplitude

% number of repetitions of the experiment
n_repetitions = 10;

% signals parameters
signal_time = 10; % secs
phase = 0;
sampling_freq = 250; % HZ
train_noise_05 = 0.7;
test_noise_05 = 0.7;
train_noise_1 = 1;
test_noise_1 = 1;
sig_len = signal_time * sampling_freq;

% final ranges of frequency and amplitude
train_frequency_range = [18.0:0.2:22.0];
test_frequency_range = [20.0];
amplitude = [1.0,3.0,5.0,10.0];

Ftr = length(train_frequency_range);
A = length(amplitude);
Fts = 1;

gauss_loglik_amp_var05 = NaN(n_repetitions, A, Ftr);
gauss_loglik_amp_var1 = NaN(n_repetitions, A, Ftr);


for n=1:n_repetitions
    
    % generate signals
    training_signals_05 = NaN(signal_time * sampling_freq,A, Ftr);
    testing_signals_05 = NaN(signal_time * sampling_freq, A);
    training_signals_1 = NaN(signal_time * sampling_freq, A,Ftr);
    testing_signals_1 = NaN(signal_time * sampling_freq, A);

    for i=1:A
        for j=1:Ftr
            training_signals_05(:,i,j) = create_signals(train_frequency_range(j), amplitude(i), signal_time, phase, sampling_freq,train_noise_05);
            training_signals_1(:,i,j) = create_signals(train_frequency_range(j), amplitude(i), signal_time, phase, sampling_freq,train_noise_1);
        end
        testing_signals_05(:,i) = create_signals(test_frequency_range, amplitude(i), signal_time, phase, sampling_freq,test_noise_05);
        testing_signals_1(:,i) = create_signals(test_frequency_range, amplitude(i), signal_time, phase, sampling_freq,test_noise_1);
    end

    % specify TDE model parameters
    lag = 21;
    
    % prepare the data to compute their autocovariance
    Y_train05 = NaN(sig_len-2*lag+1, lag*2, A,Ftr);
    Y_test05 = NaN(sig_len-2*lag+1, lag*2, A);
    sigma05 = NaN(lag*2, lag*2, A, Ftr);
    Y_train1 = NaN(sig_len-2*lag+1, lag*2, A, Ftr);
    Y_test1 = NaN(sig_len-2*lag+1, lag*2, A);
    sigma1 = NaN(lag*2, lag*2, A, Ftr);
    
    % function embeddata: makes it in the right format to compute autocovariance
    for i=1:A
        Y_test05(:,:,i) = embeddata(testing_signals_05(:,i), sig_len, [-lag+1:1:lag]);
        Y_test1(:,:,i) = embeddata(testing_signals_1(:,i), sig_len, [-lag+1:1:lag]);
        
        % train model, i.e. compute autocovariance of the signal
        for j=1:Ftr
            Y_train05(:,:,i,j) = embeddata(training_signals_05(:,i,j), sig_len, [-lag+1:1:lag]);
            sigma05(:,:,i,j) = cov(Y_train05(:,:,i,j));
       
            Y_train1(:,:,i,j) = embeddata(training_signals_1(:,i,j), sig_len, [-lag+1:1:lag]);
            sigma1(:,:,i,j) = cov(Y_train1(:,:,i,j)); 
        end
    end

    % compute loglikelihood
    for i=1:A 
        for j=1:Ftr
            gauss_loglik_amp_var05(n,i,j) = sum(logmvnpdf(Y_test05(:,:,i),0,sigma05(:,:,i,j)));
            gauss_loglik_amp_var1(n,i,j) = sum(logmvnpdf(Y_test1(:,:,i),0,sigma1(:,:,i,j)));
        end
        
        gauss_loglik_amp_var05(n,i,:) = gauss_loglik_amp_var05(n,i,:) - gauss_loglik_amp_var05(n,i,ceil(Ftr/2));
        gauss_loglik_amp_var1(n,i,:) = gauss_loglik_amp_var1(n,i,:) - gauss_loglik_amp_var1(n,i, ceil(Ftr/2));
    end
   
end


%% PLOT TDE frequency resolution varying amplitude

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];


for i=1:A
    mean1 = mean(squeeze(gauss_loglik_amp_var05(:,i,:)),1);
    mean2 = mean(squeeze(gauss_loglik_amp_var1(:,i,:)),1);
    plot(mean1,'Color',color(i, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(i, :), 'LineWidth',2);
    hold on;
    
    std_dev = std(squeeze(gauss_loglik_amp_var05(:,i,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(i,:));
    set(h,'facealpha',0.5);
    
    hold on;
    
    std_dev = std(squeeze(gauss_loglik_amp_var1(:,i,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(i,:));
    set(h,'facealpha',0.1, 'LineStyle', '-.');
    hold on;
    
end
colormap(color);
c = colorbar('Ticks', linspace(0.06,0.94,4),'TickLabels',{'1.0','3.0','5.0','7.0'}, 'Direction', 'reverse');
c.Label.String = ["amplitude"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.1];
c.Label.Rotation = 0;

lgx = legend({'noise var = 0.5', 'noise var = 1.0'},'Location', 'South');
lgx.FontSize = 16;
title('TDE', 'FontSize', 16)
xl = xlabel(' \Delta_{a} ');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;

xl = xlabel('\Delta_{f} (Hz)');
xl.FontSize = 18;
yl = ylabel('likelihood');
yl.FontSize = 18;
xticks([1:2:Ftr]);
xtickangle(45);
xticklabels({'-2.0','-1.6', '-1.2', '-0.8', '-0.4', '0.0', '0.4','0.8', '1.2','1.6', '2.0'});
xlim([1 Ftr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
grid on

%%
%% TEST TDE amplitude resolution varying frequency

% repetitions of experiment
n_repetitions = 10;

% signals parameters
signal_time = 10; % secs
phase = 0;
sampling_freq = 250; % HZ
train_noise_05 = 0.7;
test_noise_05 = 0.7;
train_noise_1 = 1;
test_noise_1 = 1;
sig_len = signal_time * sampling_freq;

% final ranges
frequency = [1,10,25,50];
train_amplitude_range = [0.5, 1.0, 2.5, 5.0,10,15, 25, 50]; % == [0.1,0.2,0.5,0.7,1.0,1.5,2.0,3.0,5.0] * test_amplitude_range
test_amplitude_range = [5.0];

F = length(frequency);
Atr = length(train_amplitude_range);
Ats = 1;

gauss_loglik_freq_var05 = NaN(n_repetitions, F, Atr);
gauss_loglik_freq_var1 = NaN(n_repetitions, F, Atr);

for n=1:n_repetitions
    % generate signals
    training_signals_05 = NaN(signal_time * sampling_freq,F, Atr);
    testing_signals_05 = NaN(signal_time * sampling_freq, F);
    training_signals_1 = NaN(signal_time * sampling_freq, F,Atr);
    testing_signals_1 = NaN(signal_time * sampling_freq, F);

    for i=1:F
        for j=1:Atr
            training_signals_05(:,i,j) = create_signals(frequency(i), train_amplitude_range(j), signal_time, phase, sampling_freq,train_noise_05);
            training_signals_1(:,i,j) = create_signals(frequency(i), train_amplitude_range(j), signal_time, phase, sampling_freq,train_noise_1);
        end
        testing_signals_05(:,i) = create_signals(frequency(i), test_amplitude_range, signal_time, phase, sampling_freq,test_noise_05);
        testing_signals_1(:,i) = create_signals(frequency(i), test_amplitude_range, signal_time, phase, sampling_freq,test_noise_1);
    end

    % set model parameter
    lag = 21;
    
    Y_train05 = NaN(sig_len-2*lag+1, lag*2, F,Atr);
    Y_test05 = NaN(sig_len-2*lag+1, lag*2, F);
    sigma05 = NaN(lag*2, lag*2, F,Atr);
    Y_train1 = NaN(sig_len-2*lag+1, lag*2, F,Atr);
    Y_test1 = NaN(sig_len-2*lag+1, lag*2, F);
    sigma1 = NaN(lag*2, lag*2, F,Atr);
    
    % embed data - train models
    for i=1:F
        Y_test05(:,:,i) = embeddata(testing_signals_05(:,i), sig_len, [-lag+1:1:lag]);
        Y_test1(:,:,i) = embeddata(testing_signals_1(:,i), sig_len, [-lag+1:1:lag]);
        
        for j=1:Atr
            Y_train05(:,:,i,j) = embeddata(training_signals_05(:,i,j), sig_len, [-lag+1:1:lag]);
            sigma05(:,:,i,j) = cov(Y_train05(:,:,i,j));

            Y_train1(:,:,i,j) = embeddata(training_signals_1(:,i,j), sig_len, [-lag+1:1:lag]);
            sigma1(:,:,i,j) = cov(Y_train1(:,:,i,j));
        end
    end

    % compute loglikelihood of train vs test signal
    for i=1:F 
        for j=1:Atr
            gauss_loglik_freq_var05(n,i,j) = sum(logmvnpdf(Y_test05(:,:,i),0,sigma05(:,:,i,j)));
            gauss_loglik_freq_var1(n,i,j) = sum(logmvnpdf(Y_test1(:,:,i),0,sigma1(:,:,i,j)));
        end
        
        gauss_loglik_freq_var05(n,i,:) = gauss_loglik_freq_var05(n,i,:) - gauss_loglik_freq_var05(n,i,ceil(Atr/2));
        gauss_loglik_freq_var1(n,i,:) = gauss_loglik_freq_var1(n,i,:) - gauss_loglik_freq_var1(n,i, ceil(Atr/2));
    end
   
end


%% PLOT TDE amplitude resolution varying frequency

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];


for f=1:F
    mean1 = mean(squeeze(gauss_loglik_freq_var05(:,f,:)),1);
    mean2 = mean(squeeze(gauss_loglik_freq_var1(:,f,:)),1);
    plot(mean1,'Color',color(f, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(f, :), 'LineWidth',2);
    hold on;
    
    std_dev = std(squeeze(gauss_loglik_freq_var05(:,f,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(f,:));
    set(h,'facealpha',0.5);
    
    hold on;
    
    std_dev = std(squeeze(gauss_loglik_freq_var1(:,f,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(f,:));
    set(h,'facealpha',0.1, 'LineStyle', '-.');
    hold on;
    
    
    %plot(log_delta_a_f_llsnr(f,:),':','Color',color(f, :), 'LineWidth',2);
end
colormap(color);
c = colorbar('Ticks', linspace(0.06,0.94,4),'TickLabels',{'1','10','25','50'}, 'Direction', 'reverse');
c.Label.String = ["frequency (Hz)"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.1];
c.Label.Rotation = 0;

lgx = legend({'noise var = 0.5', 'noise var = 1.0'},'Location', 'South');
lgx.FontSize = 16;
title('TDE', 'FontSize', 16)
xl = xlabel(' \Delta_{a} ');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;

xl = xlabel('train a = * test a');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:Atr]);
xtickangle(45);
xticklabels({'0.1','0.2','0.5','1.0','2.0','3.0','5.0', '10'});
ylim([-30000 2000]);
xlim([1 Atr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)

grid()



%%
%% TEST TDE frequency resolution varying lags

lag_list = [5,15,25,35];
n_lags = length(lag_list);

n_repetitions = 10;

% signal parameters
phase = 0;
sampling_freq = 250; % HZ
noise05 = 0.7;
noise1 = 1;
signal_time = 10; % secs
sig_len = signal_time * sampling_freq;

%ranges
train_frequency_range = [18.0:0.2:22.0];
test_frequency_range = [20.0];
amplitude = 5.0;
Ftr = length(train_frequency_range);
A = 1;
Fts = 1;

loglikgauss_lag05 = zeros(n_repetitions,n_lags,Ftr);
loglikgauss_lag1 = zeros(n_repetitions,n_lags,Ftr);

for n=1:n_repetitions
    % generate signals
    training_signals05 = zeros(signal_time * sampling_freq, Ftr);
    training_signals1 = zeros(signal_time * sampling_freq, Ftr);
    
    testing_signals05 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise05);
    testing_signals1 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise1);
    
    for i=1:Ftr
        training_signals05(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise05);
        training_signals1(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise1);
    end

    for l=1:n_lags
 
        lag = lag_list(l);
        
        Y_train05 = zeros(sig_len-2*lag+1, lag*2);
        sigma05 = zeros(lag*2, lag*2, Ftr);

        Y_train1 = zeros(sig_len-2*lag+1, lag*2);
        sigma1 = zeros(lag*2, lag*2, Ftr);
        
        % test embedded 
        Y_test05 = embeddata(testing_signals05, sig_len, [-lag+1:1:lag]);
        Y_test1 = embeddata(testing_signals1, sig_len, [-lag+1:1:lag]);
        
        for i=1:Ftr
            % embed training data
            Y_train05(:,:,i) = embeddata(training_signals05(:,i), sig_len, [-lag+1:1:lag]);
            sigma05(:,:,i) = cov(Y_train05(:,:,i));
            
            Y_train1(:,:,i) = embeddata(training_signals1(:,i), sig_len, [-lag+1:1:lag]);
            sigma1(:,:,i) = cov(Y_train1(:,:,i));
           
            % for each training signal test on y
            loglikgauss_lag05(n,l,i) = sum(logmvnpdf(Y_test05,0,sigma05(:,:,i)));
            loglikgauss_lag1(n,l,i) = sum(logmvnpdf(Y_test1,0,sigma1(:,:,i)));
            
        end
        
        loglikgauss_lag05(n,l,:) = loglikgauss_lag05(n,l,:) - loglikgauss_lag05(n,l,ceil(Ftr/2));
        loglikgauss_lag1(n,l,:) = loglikgauss_lag1(n,l,:) - loglikgauss_lag1(n,l,ceil(Ftr/2));
        
    end
end


%% PLOT TDE frequency resolution varying lags

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for l=1:n_lags
    mean1 = mean(squeeze(loglikgauss_lag05(:,l,:)),1);
    mean2 = mean(squeeze(loglikgauss_lag1(:,l,:)),1);
    plot(mean1,'Color',color(l, :), 'LineWidth',2);
    hold on;
    plot(mean2,':','Color',color(l, :), 'LineWidth',2);
    hold on;
 
    std_dev = std(squeeze(loglikgauss_lag05(:,l,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(l,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikgauss_lag1(:,l,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(l,:));
    set(h,'facealpha',0.1,'LineStyle', '-.');
    hold on;

end
title('TDE', 'FontSize', 16)
lgx = legend({'noise var = 0.5', 'noise var = 1'},'Location', 'South');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'5','15','25','35'}, 'Direction', 'reverse');
c.Label.String = "lags";
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;


xl = xlabel('\Delta_{f} (Hz)');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:2:Ftr]);
xtickangle(45);
xticklabels({'-2','-1.6', '-1.2', '-0.8', '-0.4', '0.0', '0.4','0.8', '1.2','1.6', '2'});
%ylim([-150000 2000]);
xlim([1 Ftr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)

grid()

%%
%% TEST TDE frequency resolution varying signal length

signal_time_list = [2,5,10,15];
n_time_list = length(signal_time_list);
n_repetitions = 10;

phase = 0;
sampling_freq = 250; % HZ
noise05 = 0.7;
noise1 = 1;

train_frequency_range = [18.0:0.2:22.0];
test_frequency_range = [20.0];
amplitude = 5.0;
Ftr = length(train_frequency_range);
Fts = 1;
A = 1;

loglikgauss_siglen05 = zeros(n_repetitions,n_time_list,Ftr);
loglikgauss_siglen1 = zeros(n_repetitions,n_time_list,Ftr);

for n=1:n_repetitions
    for sl=1:n_time_list
        signal_time = signal_time_list(sl); % secs

        sig_len = signal_time * sampling_freq;

        training_signals05 = zeros(signal_time * sampling_freq, Ftr);
        training_signals1 = zeros(signal_time * sampling_freq, Ftr);

        testing_signals05 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise05);
        testing_signals1 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise1);
        
        for i=1:Ftr
            training_signals05(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise05);
            training_signals1(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise1);
        end
        
        lag = 21;
        Y_train05 = zeros(sig_len-2*lag+1, lag*2, Ftr);
        sigma05 = zeros(lag*2, lag*2, Ftr);
        
        Y_train1 = zeros(sig_len-2*lag+1, lag*2, Ftr);
        sigma1 = zeros(lag*2, lag*2, Ftr);
        
        Y_test05 = embeddata(testing_signals05, sig_len, [-lag+1:1:lag]);
        Y_test1 = embeddata(testing_signals1, sig_len, [-lag+1:1:lag]);
        
        % embed data
        for i=1:Ftr
            Y_train05(:,:,i) = embeddata(training_signals05(:,i), sig_len, [-lag+1:1:lag]);
            sigma05(:,:,i) = cov(Y_train05(:,:,i));
            
            Y_train1(:,:,i) = embeddata(training_signals1(:,i), sig_len, [-lag+1:1:lag]);
            sigma1(:,:,i) = cov(Y_train1(:,:,i));
        end

        % compute loglikelihood
        for i=1:Ftr % testing
            loglikgauss_siglen05(n,sl,i) = sum(logmvnpdf(Y_test05,0,sigma05(:,:,i)));
            loglikgauss_siglen1(n,sl,i) = sum(logmvnpdf(Y_test1,0,sigma1(:,:,i)));
        end

        loglikgauss_siglen05(n,sl,:) = loglikgauss_siglen05(n,sl,:) - loglikgauss_siglen05(n,sl,ceil(Ftr/2));
        loglikgauss_siglen1(n,sl,:) = loglikgauss_siglen1(n,sl,:) - loglikgauss_siglen1(n,sl,ceil(Ftr/2));
        
    end
end


%% PLOT TDE frequency resolution varying signal length

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for sl=1:n_time_list
    mean1 = mean(squeeze(loglikgauss_siglen05(:,sl,:)),1);
    mean2 = mean(squeeze(loglikgauss_siglen1(:,sl,:)),1);
    plot(mean1,'Color',color(sl, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(sl, :), 'LineWidth',2);
 
    std_dev = std(squeeze(loglikgauss_siglen05(:,sl,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Ftr,Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikgauss_siglen1(:,sl,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.1,'LineStyle','-.');
    hold on;

end
title('TDE', 'FontSize', 16);
lgx = legend({'noise var = 0.5', 'noise var = 1'}, 'Location', 'South');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'2','5','10','15'}, 'Direction', 'reverse');
c.Label.String = ["sig-len (sec)"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('\Delta_{f} (Hz)');
xl.FontSize = 18;
yl = ylabel('likelihood');
yl.FontSize = 18;
xticks([1:2:Ftr]);
xtickangle(45);
xticklabels({'-2','-1.6', '-1.2', '-0.8', '-0.4', '0.0', '0.4','0.8', '1.2','1.6', '2'});
%ylim([-150000 2000]);
xlim([1 Ftr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
grid on


%%
%% TEST TDE amplitude resolution varying signal length

signal_time_list = [2,5,10,15];
n_time_list = length(signal_time_list);
n_repetitions = 10;

phase = 0;
sampling_freq = 250; % HZ
noise05 = 0.7;
noise1 = 1;


frequency = [20.0];
test_amplitude_range = [5.0];
train_amplitude_range = [0.5, 1.0, 2.5, 3.5, 5.0, 10,20, 50, 100]; % == [0.1,0.2,0.5,0.7,1.0,1.5,2.0,3.0,5.0] * test_amplitude_range
F = 1;
Atr = length(train_amplitude_range);
Ats = 1;

loglikgauss_siglen05 = zeros(n_repetitions,n_time_list,Atr);
loglikgauss_siglen1 = zeros(n_repetitions,n_time_list,Atr);

for n=1:n_repetitions
    for sl=1:n_time_list
        signal_time = signal_time_list(sl); % secs

        sig_len = signal_time * sampling_freq;

        training_signals05 = zeros(signal_time * sampling_freq, Atr);
        training_signals1 = zeros(signal_time * sampling_freq, Atr);

        testing_signals05 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise05);
        testing_signals1 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise1);
        
        for i=1:Atr
            training_signals05(:,i) = create_signals(frequency, train_amplitude_range(i), signal_time, phase, sampling_freq,noise05);
            training_signals1(:,i) = create_signals(frequency, train_amplitude_range(i), signal_time, phase, sampling_freq,noise1);
        end
        
        lag = 21;
        Y_train05 = zeros(sig_len-2*lag+1, lag*2, Atr);
        sigma05 = zeros(lag*2, lag*2, Atr);
        
        Y_train1 = zeros(sig_len-2*lag+1, lag*2, Atr);
        sigma1 = zeros(lag*2, lag*2, Atr);
        
        Y_test05 = embeddata(testing_signals05, sig_len, [-lag+1:1:lag]);
        Y_test1 = embeddata(testing_signals1, sig_len, [-lag+1:1:lag]);
        
        % embed data
        for i=1:Atr
            Y_train05(:,:,i) = embeddata(training_signals05(:,i), sig_len, [-lag+1:1:lag]);
            sigma05(:,:,i) = cov(Y_train05(:,:,i));
            
            Y_train1(:,:,i) = embeddata(training_signals1(:,i), sig_len, [-lag+1:1:lag]);
            sigma1(:,:,i) = cov(Y_train1(:,:,i));
        end

        % compute loglikelihood
        for i=1:Atr % testing
            loglikgauss_siglen05(n,sl,i) = sum(logmvnpdf(Y_test05,0,sigma05(:,:,i)));
            loglikgauss_siglen1(n,sl,i) = sum(logmvnpdf(Y_test1,0,sigma1(:,:,i)));
        end

        loglikgauss_siglen05(n,sl,:) = loglikgauss_siglen05(n,sl,:) - loglikgauss_siglen05(n,sl,ceil(Atr/2));
        loglikgauss_siglen1(n,sl,:) = loglikgauss_siglen1(n,sl,:) - loglikgauss_siglen1(n,sl,ceil(Atr/2));
        
    end
end


%% PLOT TDE amplitude resolution varying signal length

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for sl=1:n_time_list
    mean1 = mean(squeeze(loglikgauss_siglen05(:,sl,:)),1);
    mean2 = mean(squeeze(loglikgauss_siglen1(:,sl,:)),1);
    plot(mean1,'Color',color(sl, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(sl, :), 'LineWidth',2);
 
    std_dev = std(squeeze(loglikgauss_siglen05(:,sl,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Atr,Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikgauss_siglen1(:,sl,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.1,'LineStyle','-.');
    hold on;

end
title('TDE', 'FontSize', 16);
lgx = legend({'noise var = 0.5', 'noise var = 1'}, 'Location', 'South');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'2','5','10','15'}, 'Direction', 'reverse');
c.Label.String = ["sig-len (sec)"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('train a = * test a');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:Atr]);
xtickangle(45);
xticklabels({'0.1','0.2','0.5','0.7','1.0','2.0','5.0', '10','20'});
ylim([-70000 2000]);
xlim([1 Atr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)

grid()


%%
%% TEST TDE amplitude resolution varying lags 

lag_list = [5,15,25,35];

n_lags = length(lag_list);
n_repetitions = 10;

% signals parameters
phase = 0;
sampling_freq = 250; % HZ
noise05 = 0.7;
noise1 = 1;
signal_time = 10; % secs
sig_len = signal_time * sampling_freq;

% final ranges
frequency = [20.0];
test_amplitude_range = [5.0];
train_amplitude_range = [0.5, 1.0, 2.5, 3.5, 5.0,10, 25, 50, 100]; % == [0.1,0.2,0.5,0.7,1.0,1.5,2.0,3.0,5.0] * test_amplitude_range
F = 1;
Atr = length(train_amplitude_range);
Ats = 1;

loglikgauss_lag05 = zeros(n_repetitions,n_lags,Atr);
loglikgauss_lag1 = zeros(n_repetitions,n_lags,Atr);

for n=1:n_repetitions
    % generate signals
    training_signals05 = zeros(signal_time * sampling_freq, Atr);
    training_signals1 = zeros(signal_time * sampling_freq, Atr);
    
    testing_signals05 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise05);
    testing_signals1 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise1);
    
    for j=1:Atr
        training_signals05(:,j) = create_signals(frequency, train_amplitude_range(j), signal_time, phase, sampling_freq,noise05);
        training_signals1(:,j) = create_signals(frequency, train_amplitude_range(j), signal_time, phase, sampling_freq,noise1);
    end
    
    % for each lag config
    for l=1:n_lags
 
        lag = lag_list(l);
        
        % embed data
        Y_train05 = zeros(sig_len-2*lag+1, lag*2);
        sigma05 = zeros(lag*2, lag*2, Atr);
        Y_train1 = zeros(sig_len-2*lag+1, lag*2);
        sigma1 = zeros(lag*2, lag*2, Atr);
        
        Y_test05 = embeddata(testing_signals05, length(testing_signals05), [-lag+1:1:lag]);
        Y_test1 = embeddata(testing_signals1, sig_len, [-lag+1:1:lag]);
        
        % train model
        for i=1:Atr
            Y_train05(:,:,i) = embeddata(training_signals05(:,i), sig_len, [-lag+1:1:lag]);
            sigma05(:,:,i) = cov(Y_train05(:,:,i));
            
            Y_train1(:,:,i) = embeddata(training_signals1(:,i), sig_len, [-lag+1:1:lag]);
            sigma1(:,:,i) = cov(Y_train1(:,:,i));
           
            % test model on test data
            loglikgauss_lag05(n,l,i) = sum(logmvnpdf(Y_test05,0,sigma05(:,:,i)));
            loglikgauss_lag1(n,l,i) = sum(logmvnpdf(Y_test1,0,sigma1(:,:,i)));
        end
        
        loglikgauss_lag05(n,l,:) = loglikgauss_lag05(n,l,:) - loglikgauss_lag05(n,l,ceil(Atr/2));
        loglikgauss_lag1(n,l,:) = loglikgauss_lag1(n,l,:) - loglikgauss_lag1(n,l,ceil(Atr/2));
        
    end
end

%% PLOT TDE amplitude resolution varying lags
figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for l=1:n_lags
    mean1 = mean(squeeze(loglikgauss_lag05(:,l,:)),1);
    mean2 = mean(squeeze(loglikgauss_lag1(:,l,:)),1);
    plot(mean1,'Color',color(l, :), 'LineWidth',2);
    hold on;
    plot(mean2,':','Color',color(l, :), 'LineWidth',2);
    hold on;
 
    std_dev = std(squeeze(loglikgauss_lag05(:,l,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(l,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikgauss_lag1(:,l,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(l,:));
    set(h,'facealpha',0.1,'LineStyle', '-.');
    hold on;

end
title('TDE f=20Hz - test a = 5.0 ', 'FontSize', 16)
lgx = legend({'noise var = 0.5', 'noise var = 1'},'Location', 'South');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'5','15','25','35'}, 'Direction', 'reverse');
c.Label.String = "lags";
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('train a = * test a');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:Atr]);
xtickangle(45);
xticklabels({'0.1','0.2','0.5','0.7','1.0','2.0','5.0', '10','20'});
ylim([-60000 2000]);
xlim([1 Atr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)

grid()

