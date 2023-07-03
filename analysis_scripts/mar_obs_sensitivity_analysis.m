%% SENSITIVITY ANALYSIS ON MAR OBSERVATION MODEL

%% SCRIPT DESCRIPTION
% This script contains the code for the sensitivity analysis on the MAR 
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
%% TEST MAR frequency resolution varying amplitude

%repetitions of the experiment
n_repetitions = 10;

% parameters for signal generation
signal_time = 10; % secs
phase = 0;
sampling_freq = 250; % HZ
train_noise_var05 = 0.7; %specify noise std
test_noise_var05 = 0.7; 
train_noise_var1 = 1;
test_noise_var1 = 1;

sig_len = signal_time * sampling_freq;

%specify training and testing frequency and amplitude ranges
train_frequency_range = [19.9:.02:20.1];
amplitude_range = [1.0,3.0,5.0,7.0];
test_frequency_range = [20.0];

Ftr = length(train_frequency_range);
A = length(amplitude_range);
Fts = length(test_frequency_range);

% initialize log likelihood measure 
loglikmar_var05 = zeros(n_repetitions,A,Ftr);
loglikmar_var1 = zeros(n_repetitions,A, Ftr);


for n=1:n_repetitions
    % initialize signals
    training_signals_05 = zeros(signal_time * sampling_freq, A,Ftr);
    testing_signals_05 = zeros(signal_time * sampling_freq, A);
    training_signals_1 = zeros(signal_time * sampling_freq, A,Ftr);
    testing_signals_1 = zeros(signal_time * sampling_freq, A);
    
    % generate training signals 
    for i=1:A
        for j=1:Ftr
            training_signals_05(:,i,j) = create_signals(train_frequency_range(j), amplitude_range(i), signal_time, phase, sampling_freq,train_noise_var05);
            training_signals_1(:,i,j) = create_signals(train_frequency_range(j), amplitude_range(i), signal_time, phase, sampling_freq,train_noise_var1);
        end
        % generate test signal
        testing_signals_05(:,i) = create_signals(test_frequency_range, amplitude_range(i), signal_time, phase, sampling_freq,test_noise_var05);
        testing_signals_1(:,i) = create_signals(test_frequency_range, amplitude_range(i), signal_time, phase, sampling_freq,test_noise_var1);
    end

    %% ANALYSIS
    
    order = 3;
    maxorder = order;

    % initialize the mar components
    pred05 = zeros(sig_len-order,A,Ftr);
    residuals05 = zeros(sig_len-order,A,Ftr);
    sigma05 = zeros(A,Ftr);
    pred1 = zeros(sig_len-order,A,Ftr);
    residuals1 = zeros(sig_len-order,A,Ftr);
    sigma1 = zeros(A,Ftr);
    
    % train one mar model on each train signal
    for i=1:A
        for j=1:Ftr
            [Wml,covm ,pred05(:,i,j),residuals05(:,i,j),facerr] = mlmar(training_signals_05(:,i,j),maxorder,sig_len, [], order);
            sigma05(i,j) = std(residuals05(:,i,j));
            [Wml,covm,pred1(:,i,j),residuals1(:,i,j),facerr] = mlmar(training_signals_1(:,i,j),maxorder,sig_len, [], order);
            sigma1(i,j) = std(residuals1(:,i,j));
        end
    end
    
    % compute log likelihood of train vs test signal, 
    % for every trained model
    loglik_mar_time_05 = zeros(sig_len-order, A,Ftr);
    loglik_mar_time_1 = zeros(sig_len-order, A,Ftr);
    
    % for each time point
    for t=1:sig_len-order
        for i=1:A 
            for j=1:Ftr 
                % test on one, trained on all the others
                loglik_mar_time_05(t,i,j) = logmvnpdf(testing_signals_05(t+order,i),pred05(t,i,j),sigma05(i,j)^2);
                loglik_mar_time_1(t,i,j) = logmvnpdf(testing_signals_1(t+order,i),pred1(t,i,j),sigma1(i,j)^2);
            end
            loglik_mar_time_05(t,i,:) = loglik_mar_time_05(t,i,:) - loglik_mar_time_05(t,i,ceil(Ftr/2)); % just because test frequency is centered there
            loglik_mar_time_1(t,i,:) = loglik_mar_time_1(t,i,:) - loglik_mar_time_1(t,i,ceil(Ftr/2)); % just because test frequency is centered there
        end
    end
    % sum log likelihood over time
    loglikmar_var05(n,:,:) = sum(loglik_mar_time_05,1);
    loglikmar_var1(n,:,:) = sum(loglik_mar_time_1,1);

end

%% PLOT MAR frequency resolution varying amplitude

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];
    
j = 1;
for i=1:A 
    mean1 = mean(squeeze(loglikmar_var05(:,i,:)),1);
    mean2 = mean(squeeze(loglikmar_var1(:,i,:)),1);
    
    plot(mean1,'Color',color(j, :), 'LineWidth',2);
    hold on;
    plot(mean2,':','Color',color(j, :), 'LineWidth',2);
    
    hold on;
    std_dev = std(squeeze(loglikmar_var05(:,i,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(j,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikmar_var1(:,i,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(j,:));
    set(h,'facealpha',0.1, 'LineStyle', '-.');
    hold on;
    j = j+1;

end

colormap(color);
c = colorbar('Ticks', linspace(0.08,0.92,4),'TickLabels',{'1.0','3.0','5.0','7.0'}, 'Direction','reverse');
c.Label.String = ["amplitude"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.1];
c.Label.Rotation = 0;

lgx = legend({'noise var = 0.5', 'noise var = 1'},'Location', 'SouthWest');
lgx.FontSize = 16;
title('MAR', 'FontSize', 16)
xl = xlabel('\Delta_{f} (Hz)');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:1:Ftr]);
xtickangle(45);
xticklabels({'-0.20','-0.16','-0.12','-0.08','-0.04','0.0', '0.04', '0.08', '0.12', '0.16', '0.20'});
xlim([1 Ftr]);
%ylim([-105000 1000]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14);
grid on


%%
%% TEST MAR amplitude resolution varying frequency

% repetitions of experiment
n_repetitions = 10;

% signal parameters
signal_time = 10; % secs
phase = 0;
sampling_freq = 250; % HZ
train_noise_var05 = 0.1;
test_noise_var05 = 0.1;
train_noise_var1 = 0.5;
test_noise_var1 = 0.5;
sig_len = signal_time * sampling_freq;

% final range of amplitude and frequency
frequency_range = [1,10,25,50];
train_amplitude_range = [.5:1:10.5];
test_amplitude_range = [5.5];

F = length(frequency_range);
Atr = length(train_amplitude_range);
Ats = length(test_amplitude_range);


loglikmar_freq_var05 = zeros(n_repetitions,F, Atr);
loglikmar_freq_var1 = zeros(n_repetitions,F,Atr);

for n=1:n_repetitions

    training_signals_05 = zeros(signal_time * sampling_freq, F, Atr);
    testing_signals_05 = zeros(signal_time * sampling_freq, F);
    training_signals_1 = zeros(signal_time * sampling_freq,F, Atr);
    testing_signals_1 = zeros(signal_time * sampling_freq, F);
    
    % create signals
    for i=1:F
        for j=1:Atr
            training_signals_05(:,i,j) = create_signals(frequency_range(i), train_amplitude_range(j), signal_time, phase, sampling_freq,train_noise_var05);
            training_signals_1(:,i,j) = create_signals(frequency_range(i), train_amplitude_range(j), signal_time, phase, sampling_freq,train_noise_var1);
        end
        testing_signals_05(:,i) = create_signals(frequency_range(i), test_amplitude_range, signal_time, phase, sampling_freq,test_noise_var05);
        testing_signals_1(:,i) = create_signals(frequency_range(i), test_amplitude_range, signal_time, phase, sampling_freq,test_noise_var1);
    end

    %% mar models
    order = 3;
    maxorder = order;
    
    % initialize the mar components
    pred05 = zeros(sig_len-order,F,Atr);
    residuals05 = zeros(sig_len-order,F,Atr);
    sigma05 = zeros(F,Atr);
    
    pred1 = zeros(sig_len-order,F,Atr);
    residuals1 = zeros(sig_len-order,F,Atr);
    sigma1 = zeros(F,Atr);
    
    % train mar model for every train signal
    for i=1:F
        for j=1:Atr
            [Wml,covm,pred05(:,i,j),residuals05(:,i,j),facerr] = mlmar(training_signals_05(:,i,j),maxorder,sig_len, [], order);
            sigma05(i,j) = std(residuals05(:,i,j));

            [Wml,covm,pred1(:,i,j),residuals1(:,i,j),facerr] = mlmar(training_signals_1(:,i,j),maxorder,sig_len, [], order);
            sigma1(i,j) = std(residuals1(:,i,j));
        end
    end
    
    % compute log likelihood of train vs test signal
    loglik_mar_time_05 = zeros(sig_len-order, F, Atr);
    loglik_mar_time_1 = zeros(sig_len-order, F, Atr);
    
    for t=1:sig_len-order
        for i=1:F
            for j=1:Atr
                loglik_mar_time_05(t,i,j) = logmvnpdf(testing_signals_05(t+order,i),pred05(t,i,j),sigma05(i,j)^2);
                loglik_mar_time_1(t,i,j) = logmvnpdf(testing_signals_1(t+order,i),pred1(t,i,j),sigma1(i,j)^2);
            end
            loglik_mar_time_05(t,i,:) = loglik_mar_time_05(t,i,:) - loglik_mar_time_05(t,i,ceil(Atr/2));
            loglik_mar_time_1(t,i,:) = loglik_mar_time_1(t,i,:) - loglik_mar_time_1(t,i,ceil(Atr/2));
        end
    end
    loglikmar_freq_var05(n,:,:) = sum(loglik_mar_time_05,1);
    loglikmar_freq_var1(n,:,:) = sum(loglik_mar_time_1,1);
   
end

%% PLOT MAR amplitude resolution varying frequency

figure();
color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];


for f=1:length(color)
    mean1 = mean(squeeze(loglikmar_freq_var05(:,f,:)),1);
    mean2 = mean(squeeze(loglikmar_freq_var1(:,f,:)),1);
    plot(mean1,'Color',color(f, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(f, :), 'LineWidth',2);
    hold on;
    
    std_dev = std(squeeze(loglikmar_freq_var05(:,f,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(f,:));
    set(h,'facealpha',0.5);
    
    hold on;
    
    std_dev = std(squeeze(loglikmar_freq_var1(:,f,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(f,:));
    set(h,'facealpha',0.1, 'LineStyle', '-.');
    hold on;
    
end
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'1','10', '25', '50'}, 'Direction', 'reverse');
c.Label.String = ["frequency (Hz)"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.1];
c.Label.Rotation = 0;

lgx = legend({'noise var = 0.5', 'noise var = 1'},'Location', 'South');
lgx.FontSize = 16;
title('MAR', 'FontSize', 16)
xl = xlabel(' \Delta_{a}');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;

xticks([1:1:Atr]);
xtickangle(45);
xticklabels({'-5.0','-4.0','-3.0','-2.0','-1.0','0.0','1.0','2.0','3.0','4.0','5.0'});
xlim([1 Atr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
%ylim([-5000 300])
grid on


%%
%% TEST MAR frequency resolution varying signal length

n_repetitions = 10;
signal_time_list = [2,5,10,15];
n_signal_time = length(signal_time_list);

phase = 0;
sampling_freq = 250; % HZ

train_frequency_range = [19.5:0.02:20.5];
test_frequency_range = [20.0];
amplitude = 5.0;
noise05 = 0.7; 
noise1 = 1; 

Ftr = length(train_frequency_range);
Fts = 1;
A = 1;

loglikmar_siglen05 = zeros(n_repetitions,n_signal_time,Ftr);
loglikmar_siglen1 = zeros(n_repetitions,n_signal_time,Ftr);


for n=1:n_repetitions

    for sl=1:n_signal_time
        signal_time = signal_time_list(sl); % secs
        sig_len = signal_time * sampling_freq;
        
        % test signal
        testing_signals05 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise05);
        testing_signals1 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise1);
        
        % training signals
        training_signals05 = zeros(signal_time * sampling_freq, Ftr);
        training_signals1 = zeros(signal_time * sampling_freq, Ftr);
        for i=1:Ftr
            training_signals05(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise05);
            training_signals1(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise1);
        end

        %% mar models
        order = 3;
        maxorder = order;
        
        % initialize the mar components
        pred05 = zeros(sig_len-order,Ftr);
        residuals05 = zeros(sig_len-order,Ftr);
        sigma05 = zeros(Ftr,1);
        
        pred1 = zeros(sig_len-order,Ftr);
        residuals1 = zeros(sig_len-order,Ftr);
        sigma1 = zeros(Ftr,1);

        % train models
        for i=1:(Ftr)
            [Wml,covm,pred05(:,i),residuals05(:,i),facerr] = mlmar(training_signals05(:,i),maxorder,sig_len, [], order);
            sigma05(i) = std(residuals05(:,i));
            [Wml,covm,pred1(:,i),residuals1(:,i),facerr] = mlmar(training_signals1(:,i),maxorder,sig_len, [], order);
            sigma1(i) = std(residuals1(:,i));
        end

        % compute log likelihood 
        loglik_mar_time05 = zeros(sig_len-order,Ftr);
        loglik_mar_time1 = zeros(sig_len-order, Ftr);
        
        for t=1:sig_len-order
            for i=1:Ftr
                loglik_mar_time05(t,i) = logmvnpdf(testing_signals05(t+order),pred05(t,i),sigma05(i)^2);
                loglik_mar_time1(t,i) = logmvnpdf(testing_signals1(t+order),pred1(t,i),sigma1(i)^2);
            end
            
            loglik_mar_time05(t,:) = loglik_mar_time05(t,:) - loglik_mar_time05(t,ceil(Ftr/2));
            loglik_mar_time1(t,:) = loglik_mar_time1(t,:) - loglik_mar_time1(t,ceil(Ftr/2));
        end
        
        loglikmar_siglen05(n,sl,:) = sum(loglik_mar_time05,1);
        loglikmar_siglen1(n,sl,:) = sum(loglik_mar_time1,1);
        
    end
end

%% PLOT MAR frequency resolution varying signal length

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for sl=1:n_signal_time
    mean1 = mean(squeeze(loglikmar_siglen05(:,sl,:)),1);
    mean2 = mean(squeeze(loglikmar_siglen1(:,sl,:)),1);
    plot(mean1,'Color',color(sl, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(sl, :), 'LineWidth',2);
 
    std_dev = std(squeeze(loglikmar_siglen05(:,sl,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikmar_siglen1(:,sl,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.1,'LineStyle','-.');
    hold on;

end
title('MAR', 'FontSize', 16)
lgx = legend({'noise var = 0.5','noise var = 1'}, 'Location', 'SouthEast');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.06,0.94,4),'TickLabels',{'2','5','10','15'}, 'Direction', 'reverse');
c.Label.String = ["sig-len (sec)"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('\Delta_{f} (Hz)');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:5:Ftr]);
xtickangle(45);
xticklabels({'-0.5','-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1','0.2', '0.3','0.4', '0.5'});
%ylim([-30000 1000]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
xlim([1 Ftr]);
grid on


%%
%% TEST MAR amplitude resolution varying signal length

n_repetitions = 10;
signal_time_list = [2,5,10,15];
n_signal_time = length(signal_time_list);

phase = 0;
sampling_freq = 250; % HZ

frequency = [20];
train_amplitude_range = [0.5:1:10.5];
test_amplitude_range = [5.5];
noise05 = 0.7; 
noise1 = 1; 

F = 1;
Atr = length(train_amplitude_range);
Ats = 1;

loglikmar_siglen05 = zeros(n_repetitions,n_signal_time,Atr);
loglikmar_siglen1 = zeros(n_repetitions,n_signal_time,Atr);

for n=1:n_repetitions

    for sl=1:n_signal_time
        signal_time = signal_time_list(sl); 
        sig_len = signal_time * sampling_freq; 

        % generate test signal
        testing_signals05 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise05);
        testing_signals1 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise1);
        
        % generate training signals
        training_signals05 = zeros(signal_time * sampling_freq, Atr);
        training_signals1 = zeros(signal_time * sampling_freq, Atr);
        for i=1:Atr
            training_signals05(:,i) = create_signals(frequency, train_amplitude_range(i), signal_time, phase, sampling_freq,noise05);
            training_signals1(:,i) = create_signals(frequency, train_amplitude_range(i), signal_time, phase, sampling_freq,noise1);
        end

        order = 3;
        maxorder = order;
        
        % initialize the mar components
        pred05 = zeros(sig_len-order,Atr);
        residuals05 = zeros(sig_len-order,Atr);
        sigma05 = zeros(Atr,1);
        
        pred1 = zeros(sig_len-order,Atr);
        residuals1 = zeros(sig_len-order,Atr);
        sigma1 = zeros(Atr,1);

        % train mar models
        for i=1:(Atr)
            [Wml,covm,pred05(:,i),residuals05(:,i),facerr] = mlmar(training_signals05(:,i),maxorder,sig_len, [], order);
            sigma05(i) = std(residuals05(:,i));
            [Wml,covm,pred1(:,i),residuals1(:,i),facerr] = mlmar(training_signals1(:,i),maxorder,sig_len, [], order);
            sigma1(i) = std(residuals1(:,i));
        end

        % compute log likelihood 
        loglik_mar_time05 = zeros(sig_len-order, Atr);
        loglik_mar_time1 = zeros(sig_len-order, Atr);
        
        for t=1:sig_len-order
            for i=1:Atr % trained signal
                loglik_mar_time05(t,i) = logmvnpdf(testing_signals05(t+order),pred05(t,i),sigma05(i)^2);
                loglik_mar_time1(t,i) = logmvnpdf(testing_signals1(t+order),pred1(t,i),sigma1(i)^2);
            end
            
            loglik_mar_time05(t,:) = loglik_mar_time05(t,:) - loglik_mar_time05(t,ceil(Atr/2));
            loglik_mar_time1(t,:) = loglik_mar_time1(t,:) - loglik_mar_time1(t,ceil(Atr/2));
        end
        
        loglikmar_siglen05(n,sl,:,:) = sum(loglik_mar_time05,1);
        loglikmar_siglen1(n,sl,:,:) = sum(loglik_mar_time1,1);
      
    end
end

%% PLOT MAR amplitude resolution varying signal length

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for sl=1:n_signal_time
    mean1 = mean(squeeze(loglikmar_siglen05(:,sl,:)),1);
    mean2 = mean(squeeze(loglikmar_siglen1(:,sl,:)),1);
    plot(mean1,'Color',color(sl, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(sl, :), 'LineWidth',2);
 
    std_dev = std(squeeze(loglikmar_siglen05(:,sl,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikmar_siglen1(:,sl,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(sl,:));
    set(h,'facealpha',0.1,'LineStyle','-.');
    hold on;

end
title('MAR', 'FontSize', 16)
lgx = legend({'noise var = 0.5','noise var = 1'}, 'Location', 'SouthWest');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'2','5','10','15'}, 'Direction', 'reverse');
c.Label.String = ["sig-len (sec)"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('\Delta_{a}');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:Atr]);
xtickangle(45);
xticklabels({'-5.0','-4.0','-3.0','-2.0','-1.0','0.0','1.0','2.0','3.0','4.0','5.0'});
%ylim([-150000 2000]);
xlim([1 Atr]);
%ylim([-5000 500]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
grid()


%%
%% TEST MAR frequency resolution varying order

n_repetitions = 10;
orders_list = [3,5,7,9];
n_orders = length(orders_list);

phase = 0;
sampling_freq = 250; % HZ

train_frequency_range = [19.5:0.02:20.5];
test_frequency_range = [20.0];
amplitude = 5.0;
noise05 = 0.7; 
noise1 = 1; 

Ftr = length(train_frequency_range);
Fts = 1;
A = 1;

signal_time = 10; % secs
sig_len = signal_time * sampling_freq;

loglikmar_orders_05 = zeros(n_repetitions,n_orders,Ftr);
loglikmar_orders_1 = zeros(n_repetitions,n_orders,Ftr);


for n=1:n_repetitions
    
    training_signals05 = zeros(signal_time * sampling_freq, Ftr);
    training_signals1 = zeros(signal_time * sampling_freq,Ftr);

    testing_signals05 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise05);
    testing_signals1 = create_signals(test_frequency_range, amplitude, signal_time, phase, sampling_freq,noise1);
    
    for i=1:Ftr
        training_signals05(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise05);
        training_signals1(:,i) = create_signals(train_frequency_range(i), amplitude, signal_time, phase, sampling_freq,noise1);
    end
    
    for o=1:n_orders
        
        order = orders_list(o);
        maxorder = order;
        % initialize the mar components
        pred05 = zeros(sig_len-order,Ftr);
        residuals05 = zeros(sig_len-order,Ftr);
        sigma05 = zeros(Ftr,1);
        
        pred1 = zeros(sig_len-order,Ftr);
        residuals1 = zeros(sig_len-order,Ftr);
        sigma1 = zeros(Ftr,1);

        % compute mar model for every signal
        for i=1:Ftr
            [Wml,covm,pred05(:,i),residuals05(:,i),facerr] = mlmar(training_signals05(:,i),maxorder,sig_len, [], order);
            sigma05(i) = std(residuals05(:,i));
            [Wml,covm,pred1(:,i),residuals1(:,i),facerr] = mlmar(training_signals1(:,i),maxorder,sig_len, [], order);
            sigma1(i) = std(residuals1(:,i));
        end

        % compute log likelihood of every signal belonging to the model

        loglik_mar_time05 = zeros(sig_len-order, Ftr);
        loglik_mar_time1 = zeros(sig_len-order, Ftr);
        
        for t=1:sig_len-order
            for i=1:Ftr
                loglik_mar_time05(t,i) = logmvnpdf(testing_signals05(t+order),pred05(t,i),sigma05(i)^2);
                loglik_mar_time1(t,i) = logmvnpdf(testing_signals1(t+order),pred1(t,i),sigma1(i)^2);
            end
                % subtract the ii element 
            loglik_mar_time05(t,:) = loglik_mar_time05(t,:) - loglik_mar_time05(t,ceil(Ftr/2));
            loglik_mar_time1(t,:) = loglik_mar_time1(t,:) - loglik_mar_time1(t,ceil(Ftr/2));
        end
        
        loglikmar_orders_05(n,o,:) = sum(loglik_mar_time05,1);
        loglikmar_orders_1(n,o,:) = sum(loglik_mar_time1,1);
        
    end
end

%% PLOT MAR frequency resolution varying order

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];
    %0.9 0 0.1];

for o=1:n_orders
    mean1 = mean(squeeze(loglikmar_orders_05(:,o,:)),1);
    mean2 = mean(squeeze(loglikmar_orders_1(:,o,:)),1);
    plot(mean1,'Color',color(o, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(o, :), 'LineWidth',2);
 
    std_dev = std(squeeze(loglikmar_orders_05(:,o,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(o,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikmar_orders_1(:,o,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Ftr, Ftr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(o,:));
    set(h,'facealpha',0.1,'LineStyle','-.');
    hold on;

end
title('MAR', 'FontSize', 16)
lgx = legend({'noise var = 0.5','noise var = 1'}, 'Location', 'SouthEast');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'3','5','7','9'}, 'Direction', 'reverse');
c.Label.String = ["order"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('\Delta_{f} (Hz)');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:5:Ftr]);
xtickangle(45);
xticklabels({'-0.5','-0.4', '-0.3', '-0.2', '-0.1', '0.0', '0.1','0.2', '0.3','0.4', '0.5'});
%ylim([-28000 1000]);
xlim([1 Ftr]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
grid()

%%
%% TEST MAR amplitude resolution varying order

n_repetitions = 10;
orders_list = [3,5,7,9];
n_orders = length(orders_list);

phase = 0;
sampling_freq = 250; % HZ

frequency = [20.0];
train_amplitude_range = [0.5:1:10.5];
test_amplitude_range = [5.5];
noise05 = 0.7; 
noise1 = 1; 

F = 1;
Atr = length(train_amplitude_range);
Ats = 1;

signal_time = 10; % secs
sig_len = signal_time * sampling_freq;

loglikmar_orders_05 = zeros(n_repetitions,n_orders,Atr);
loglikmar_orders_1 = zeros(n_repetitions,n_orders,Atr);

for n=1:n_repetitions
    
    training_signals05 = zeros(signal_time * sampling_freq, Atr);
    training_signals1 = zeros(signal_time * sampling_freq, Atr);
    
    testing_signals05 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise05);
    testing_signals1 = create_signals(frequency, test_amplitude_range, signal_time, phase, sampling_freq,noise1);

    for i=1:Atr
        training_signals05(:,i) = create_signals(frequency, train_amplitude_range(i), signal_time, phase, sampling_freq,noise05);
        training_signals1(:,i) = create_signals(frequency, train_amplitude_range(i), signal_time, phase, sampling_freq,noise1);
    end
    
    for o=1:n_orders
        order = orders_list(o);
        maxorder = order;
        % initialize the mar components
        pred05 = zeros(sig_len-order,Atr);
        residuals05 = zeros(sig_len-order,Atr);
        sigma05 = zeros(Atr,1);
        
        pred1 = zeros(sig_len-order,Atr);
        residuals1 = zeros(sig_len-order,Atr);
        sigma1 = zeros(Atr,1);

        % compute mar model for every signal
        for i=1:Atr
            [Wml,covm,pred05(:,i),residuals05(:,i),facerr] = mlmar(training_signals05(:,i),maxorder,sig_len, [], order);
            sigma05(i) = std(residuals05(:,i));
            [Wml,covm,pred1(:,i),residuals1(:,i),facerr] = mlmar(training_signals1(:,i),maxorder,sig_len, [], order);
            sigma1(i) = std(residuals1(:,i));
        end

        % compute log likelihood
        loglik_mar_time05 = zeros(sig_len-order, Atr);
        loglik_mar_time1 = zeros(sig_len-order, Atr);
        
        for t=1:sig_len-order
            for i=1:Atr
                loglik_mar_time05(t,i) = logmvnpdf(testing_signals05(t+order),pred05(t,i),sigma05(i)^2);
                loglik_mar_time1(t,i) = logmvnpdf(testing_signals1(t+order),pred1(t,i),sigma1(i)^2);
            end
            loglik_mar_time05(t,:) = loglik_mar_time05(t,:) - loglik_mar_time05(t, ceil(Atr/2));
            loglik_mar_time1(t,:) = loglik_mar_time1(t,:) - loglik_mar_time1(t,ceil(Atr/2));

        end
        
        loglikmar_orders_05(n,o,:) = sum(loglik_mar_time05,1);
        loglikmar_orders_1(n,o,:) = sum(loglik_mar_time1,1);

    end
end

%% PLOT MAR amplitude resolution varying order

figure();

color = [0 0.2 0.8
    %0.2 0.7 0.9
    0 0.5 0
    0.9 0.9 0.1
    1 0.5 0.1];

for o=1:n_orders
    mean1 = mean(squeeze(loglikmar_orders_05(:,o,:)),1);
    mean2 = mean(squeeze(loglikmar_orders_1(:,o,:)),1);
    plot(mean1,'Color',color(o, :), 'LineWidth',2);
    hold on
    plot(mean2,':','Color',color(o, :), 'LineWidth',2);
 
    std_dev = std(squeeze(loglikmar_orders_05(:,o,:)));
    curve1 = mean1 + std_dev;
    curve2 = mean1 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(o,:));
    set(h,'facealpha',0.5);
    hold on;
    
    std_dev = std(squeeze(loglikmar_orders_1(:,o,:)));
    curve1 = mean2 + std_dev;
    curve2 = mean2 - std_dev;
    x = linspace(1,Atr, Atr);
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    h = fill(x2, inBetween,color(o,:));
    set(h,'facealpha',0.1,'LineStyle','-.');
    hold on;

end
title('MAR', 'FontSize', 16)
lgx = legend({'noise var = 0.5','noise var = 1'}, 'Location', 'South');
lgx.FontSize = 16;
colormap(color);
c = colorbar('Ticks', linspace(0.07,0.93,4),'TickLabels',{'3','5','7','9'}, 'Direction', 'reverse');
c.Label.String = ["order"];
c.Label.FontSize = 16;
c.Label.FontWeight= 'bold';
c.Label.FontAngle= 'italic';
c.Label.Position = [0.7 -0.08];
c.Label.Rotation = 0;

xl = xlabel('\Delta_{a}');
xl.FontSize = 18;
yl = ylabel('log likelihood');
yl.FontSize = 18;
xticks([1:Atr]);
xtickangle(45);
xticklabels({'-5.0','-4.0','-3.0','-2.0','-1.0','0.0','1.0','2.0','3.0','4.0','5.0'});
%ylim([-150000 2000]);
xlim([1 Atr]);
%ylim([-5000 500]);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',14)
grid()




