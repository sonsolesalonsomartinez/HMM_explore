
%% TEST HMM INFERENCE - ONE CHANNEL, ONE FREQUENCY CONTENT

%% SCRIPT DESCRIPTION
% This script contains the code for testing HMM inference, on one/channel, 
% non stationary data with one frequency content per time point, producing
% results in Figure 3 and Supplementary Figure 2. 
% By modifying the ranges tested can be reproduced also Supplementary 
% Figure 6 and 7. 
%
% last part of the script has been used for Supplementary Figure 4.

%% ANALYSIS 
% Signals are generated as a long sequence of 50000 points and are then cut
% into 100 trials of 500 time points each for the regression analysis,
% predicting frequency or amplitude from the states time courses.

%% AUTHOR
% Laura Masaracchia 
% lauramacfin.au.dk
% Aarhus University
% June 2023

%% TEST HMM on non stationary signals: predict frequency and amplitude of
%% a) signals changing only in power - small frequency range
%% b) signals changing both in frequency and power

%% CHANGE STRING TO DECIDE ON THE ANALYSIS

ANALYSIS_TYPE = 'a';

if ANALYSIS_TYPE == 'a'
    K=4;
    % set signals parameters
    spont_options = struct();
    spont_options.POW_AR_W = 0.999; %slowly changing
    spont_options.FREQ_AR_W = 0.999;
    spont_options.POW_RANGE = [0.1 10.0];
    spont_options.FREQ_RANGE =[0.01 0.99]; % from 1 to 45 Hz
    spont_options.NOISE = 1;
    
    % set plotting parameters
    plot_d_ylim =[0 1];
    plot_d_yticks =[0 0.5 1.0];
    plot_d_yticklabels={'0','20','40'};
    
    plot_ps_xlim = [0 180];
    plot_ps_xticks = [0:20:180];
    plot_ps_xticklabels={'0','5','10','15','20','25','30','35','40','45'};
    
    
elseif ANALYSIS_TYPE =='b'
    K=2;
    % set signals parameters
    spont_options = struct();
    spont_options.POW_AR_W = 0.999; % varying slowly
    spont_options.FREQ_AR_W = 1; % almost static
    spont_options.POW_RANGE = [0.1 10.0];
    spont_options.FREQ_RANGE =[0.101 0.102]; % between 4 and 5 Hz
    spont_options.NOISE = 1; % noise var
    
    % set plotting parameters
    plot_d_ylim =[0.09 0.12];
    plot_d_yticks =[0.099 0.104];
    plot_d_yticklabels={'4.0','5.0'};
    
    plot_ps_xlim =[0 36];
    plot_ps_xticks= [0:4:36];
    plot_ps_xticklabels={'0','1','2','3','4','5','6','7','8','9'};
end
    
%% SET ALL OTHER PARAMETERS

% number repetitions of experiment
n_repetitions = 10;

% number permutations for statistical testing
n_permutations = 10000;

% regression parameters
n_trials = 100;
time_points_per_trial = 500;

% HMM parameters
T = n_trials*time_points_per_trial;
Fs = 250;

% ranges to test
dirdiag_list = [100,1000,10000,100000,1000000,10000000, 100000000, 1000000000];
lags_list= [5,9,15,21,50]; % L 
steps_list = [1,1,3,3,10]; % S
order_list = [3,5,7,9]; % P
n_order = length(order_list);
n_lags = length(lags_list);
n_dird = length(dirdiag_list);

regr_explv_pow_mar = NaN(n_repetitions, n_dird, n_order);
regr_explv_pow_rand = NaN(n_permutations,1);
regr_explv_pow_tde = NaN(n_repetitions, n_dird, n_lags);
regr_explv_freq_mar = NaN(n_repetitions, n_dird, n_order);
regr_explv_freq_rand = NaN(n_permutations,1);
regr_explv_freq_tde = NaN(n_repetitions, n_dird, n_lags);
switch_rate_mar = NaN(n_repetitions, n_dird, n_order);
switch_rate_tde = NaN(n_repetitions, n_dird, n_lags);


%% RUN ANALYSES
% manipulating delta, dirichlet diagonal, that affects the switching rate
for dr =  1:n_dird

    for g=1:n_repetitions
        % create signals
        [data,Freq,Power,spont_options,evoked_options] = simsignal(T,1,spont_options,[],[],0);
        
        %% run HMM-MAR on signals changing both pow and freq
        % build configs to test
        template_configuration = struct(); 
        configurations_mar = {}; % parameters of the HMM
        for o = 1:n_order
            configurations_mar{o} = template_configuration;
            configurations_mar{o}.order = order_list(o);
            configurations_mar{o}.K = K;
            configurations_mar{o}.covtype = 'uniquediag';
            configurations_mar{o}.useParallel=0;
            configurations_mar{o}.DirichletDiag = dirdiag_list(dr);
            configurations_mar{o}.Fs = Fs; % Sampling rate 
            configurations_mar{o}.standardise = 0; % Sampling rate 
            configurations_mar{o}.fpass = [0 50];  % band of frequency you're interested in
            configurations_mar{o}.p = 0; % interval of confidence  
            configurations_mar{o}.to_do = [1 1];
            configurations_mar{o}.Nf = 120; 
        end

        Gamma_mar = cell(n_order,1);
        hmm_mar = cell(n_order,1);
        padded_gamma_mar = cell(n_order,1);
        spectra_mar = cell(n_order,1);
        
        % run HMM-MAR
        for o = 1:n_order 
            [hmm_mar{o},Gamma_mar{o}] = hmmmar(data,T,configurations_mar{o});
            padded_gamma_mar{o} = padGamma(Gamma_mar{o},T,configurations_mar{o});
            spectra_mar{o} = hmmspectramt(data,T,Gamma_mar{o},configurations_mar{o});
            switch_rate_mar(g,dr,o) =getSwitchingRate(Gamma_mar{o},T,configurations_mar{o});
            
            % predict power and frequency from state tie courses
            T1 = ones(n_trials,1)*time_points_per_trial;
            options = {};
            [regr_explv_pow_mar(g,dr,o), beta_mar_pow] = ridge_regression_cv(padded_gamma_mar{o},Power,T1, options);
            [regr_explv_freq_mar(g,dr,o), beta_mar_freq] = ridge_regression_cv(padded_gamma_mar{o},Freq,T1, options);

        end


        %% run HMM-TDE
        
        % build HMM-TDE configs
        config_temp_tde = struct();
        configurations_tde = {}; % parameters of the HMM

        for l=1:n_lags
            configurations_tde{l} = config_temp_tde;
            configurations_tde{l}.K = K;
            configurations_tde{l}.useParallel=0;
            configurations_tde{l}.DirichletDiag =dirdiag_list(dr);
            configurations_tde{l}.embeddedlags = [-lags_list(l):steps_list(l):lags_list(l)];
            configurations_tde{l}.Fs = Fs;
            configurations_tde{l}.fpass = [0 50];
            configurations_tde{l}.p = 0;
            configurations_tde{l}.to_do = [1 1];
            configurations_tde{l}.Nf = 120;
        end

        Gamma_tde = cell(n_lags,1);
        hmm_tde = cell(n_lags,1);
        padded_gamma_tde = cell(n_lags,1);
        spectra_tde = cell(n_lags,1);
        
        % run HMM-TDE
        for l = 1:n_lags
            [hmm_tde{l},Gamma_tde{l}] = hmmmar(data,T,configurations_tde{l});
            padded_gamma_tde{l} = padGamma(Gamma_tde{l},T,configurations_tde{l});
            spectra_tde{l} = hmmspectramt(data,T,Gamma_tde{l},configurations_tde{l});
            switch_rate_tde(g,dr,l) =getSwitchingRate(Gamma_tde{l},T,configurations_tde{l}); 
            
            % predict power and frequency from states
            [regr_explv_pow_tde(g,dr,l), beta_tde_pow] = ridge_regression_cv(padded_gamma_tde{l},Power,T1, options);
            [regr_explv_freq_tde(g,dr,l), beta_tde_freq] = ridge_regression_cv(padded_gamma_tde{l},Freq,T1, options);
        end

    end
end

%% Statistical testing 
% how well can random states predict accuracy?
Yp = reshape(Power,[n_trials,time_points_per_trial]);
Yf = reshape(Freq, [n_trials,time_points_per_trial]);

%permute the labels of trials
for s=1:n_permutations    
    r = randperm(n_trials); 
    randomPow = Yp(r,:);
    randomFreq = Yf(r,:);
    randomPow = reshape(randomPow,[n_trials*time_points_per_trial,1]);
    randomFreq = reshape(randomFreq,[n_trials*time_points_per_trial,1]);
    [regr_explv_pow_rand(s), rand_beta_pow] = ridge_regression_cv(padded_gamma_tde{1},randomPow,T1,options);
    [regr_explv_freq_rand(s), rand_beta_freq] = ridge_regression_cv(padded_gamma_tde{1},randomFreq,T1,options);
 
end

% compute mean and standard deviation of random predictions
mean_rand_pow_pred = mean(regr_explv_pow_rand);
std_rand_pow_pred = std(regr_explv_pow_rand);
mean_rand_freq_pred = mean(regr_explv_freq_rand);
std_rand_freq_pred = std(regr_explv_freq_rand);


%% plot data example
% last generated data

figure();
subplot(3,1,1);
plotting_range = [1*Fs:15*Fs];
plot(data(plotting_range), 'k');
xticks([Fs:3*Fs:24*Fs]);
xticklabels({'', '','','', '', '', ''});
xlim([0,13*Fs]);

subplot(3,1,2);
plot(Freq(plotting_range), 'k', 'LineWidth', 2);
xticks([Fs:3*Fs:24*Fs]);
xticklabels({''});
xlim([0,13*Fs]);
ylabel('inst freq (Hz)');
ylim(plot_d_ylim);
yticks(plot_d_yticks);
yticklabels(plot_d_yticklabels);

subplot(3,1,3);
plot(Power(plotting_range), 'k', 'LineWidth',2);
xlabel('time (s)');
xlim([0,13*Fs]);
xticks([Fs:3*Fs:24*Fs]);
xticklabels({'1', '4','7','10', '13'})
ylabel('amplitude');

sgtitle('example of signal ');

 %% plot states probability time courses 

plotting_range = [1*Fs:15*Fs];

plx = data - min(data);
plx = plx/max(plx);

% for the last delta, plot first rep
for i=1
    figure();
    subplot(2,1,1);
    area(padded_gamma_mar{i}(plotting_range,:));
    ylabel('P(state)')
    hold on;
    plot(plx(plotting_range),'k');
    xticks([Fs:3*Fs:24*Fs]);
    xticklabels({'1', '4','7','10', '13','16','19','22'})
    xlim([0,13*Fs]);
    ylim([0 1]);
    xlabel('time (s)');
    title('HMM-MAR states probability', 'FontSize',14);
    subplot(2,1,2);
    area(padded_gamma_tde{i}(plotting_range,:));
    ylabel('P(state)')
    hold on;
    plot(plx(plotting_range),'k');
    xticks([Fs:3*Fs:14*Fs]);
    xticklabels({'1', '4','7','10', '13'})
    xlim([0,13*Fs]);
    ylim([0,1]);
    xlabel('time (s)');
    title('HMM-TDE states probability', 'FontSize',14);

end
%% compute and plot states power spectra

% for each state, only the first repetition
for k=1:K
    mar_state_pow_ch1(:,k) = spectra_mar{1}.state(k).psd(:,1,1);
end


for k=1:K
    tde_state_pow_ch1(:,k) = spectra_tde{1}.state(k).psd(:,1,1);
end

figure();
plot(mar_state_pow_ch1, 'Linewidth',2);
xlim(plot_ps_xlim)
xticks(plot_ps_xticks);
xticklabels(plot_ps_xticklabels);
xlabel('frequency (Hz)');
lgd = legend({'1', '2','3','4','5','6','7','8','9','10'});
lgd.Title.String = 'states';
lgd.FontSize = 14;
grid()
ylabel('power')
title('HMM-MAR frequency content of states', 'FontSize', 14);


figure();
plot(tde_state_pow_ch1, 'Linewidth',2);
xlim(plot_ps_xlim);
xticks(plot_ps_xticks);
xticklabels(plot_ps_xticklabels);
xlabel('frequency (Hz)');
ylabel('power')
grid()
lgd = legend({'1', '2','3','4','5','6','7','8','9','10'});
lgd.Title.String = 'states';
lgd.FontSize = 14;
title('HMM-TDE power content of states', 'FontSize',14);


%% 3D plot of prediction accuracy
%as function of switching rate and hyperparameters 
% !!!! PLOT EITHER ACCURACY ON AMPLITUDE ORON FREQUENCY !!!!!!!!

%show prediction on amplitude
%Z_TDE = squeeze(mean(regr_explv_pow_tde,1));
%Z_MAR = squeeze(mean(regr_explv_pow_mar,1));

%show prediction on frequency
Z_TDE = squeeze(mean(regr_explv_freq_tde,1));
Z_MAR = squeeze(mean(regr_explv_freq_mar,1));


[X1,Y1] = meshgrid(lags_list, dirdiag_list);

figure()
surf(X1,Y1,Z_TDE);
colormap autumn

xlabel('lags')
ylabel('\delta')
zlabel('CVEV')
title('predict amplitude HMM-TDE')
%title('predict frequency HMM-TDE')


[X2,Y2] = meshgrid(order_list, dirdiag_list);

figure()
surf(X2,Y2,Z_MAR);
colormap winter
xlabel('order')
ylabel('\delta')
zlabel('CVEV')
%title('predict amplitude HMM-MAR')
title('predict frequency HMM-TDE')

%% 2D PLOT of prediction accuracy 
%as a function of switching rate
% !!!!!!! SHOW EITHER PREDICTION ON FREQUENCY OR ON AMPLITUDE !!!!!!!!

% show prediction on amplitude
%y_TDE = squeeze(regr_explv_pow_tde(:,dr,i))';
%y_MAR = squeeze(regr_explv_pow_tde(:,dr,i))';

%show prediction on frequency
y_TDE = squeeze(regr_explv_freq_tde(:,dr,i))';
y_MAR = squeeze(regr_explv_freq_tde(:,dr,i))';

figure();

% for one fixed config of lags and order
i = 2;
c_TDE = colormap(autumn(n_dird));
c_MAR = colormap(winter(n_dird)); 

for dr=1:n_dird
    scatter(squeeze(switch_rate_tde(:,dr,i))',y_TDE,50,c_TDE(dr,:),'filled');
    hold on
    scatter(squeeze(switch_rate_mar(:,dr,i))',y_MAR,50,c_MAR(dr,:),'filled');
    hold on
   
end

%title('predict amplitude from states');
title('predict frequency from states');
xlabel('states switching rate');
ylabel('CVEV');
grid on

%% 3D PLOT of switching rate as function of delta and hyperparameters

Z_TDE = squeeze(mean(switch_rate_tde,1));
Z_MAR = squeeze(mean(switch_rate_mar,1));

[X1,Y1] = meshgrid(lags_list, dirdiag_list);

figure()
surf(X1,Y1,Z_TDE);
colormap autumn

xlabel('lags')
ylabel('Dirichlet Diagonal')
zlabel('switch rate')
title('states switching rate HMM-TDE')


[X2,Y2] = meshgrid(order_list, dirdiag_list);

figure()
surf(X2,Y2,Z_MAR);
colormap winter
xlabel('order')
ylabel('Dirichlet Diagonal')
zlabel('switch rate')
title('states switching rate HMM-MAR')


%% 3D scatter plot of prediction accuracy 
% as function of dirdiag and hyperp
%!!!!! SHOW EITHER PREDICTION OF AMPLITUDE OR OF FREQUENCY !!!!!

figure();
for dr=1:n_dird
    for i = 1:n_lags
        for g=1:n_repetitions
            %show prediction of amplitude
            %scatter3(dirdiag_list(dr),lags_list(i), regr_explv_pow_tde(g,dr,i),20,'r','filled');
            %show prediction of frequency
            scatter3(dirdiag_list(dr),lags_list(i), regr_explv_freq_tde(g,dr,i),20,'r','filled');
            hold on
        end
    end
end

%title('predict amplitude HMM-TDE');
title('predict frequency HMM-TDE');

xlabel('Dirdiag');
ylabel('lags');
zlabel('CVEV')


figure();
for dr=1:n_dird
    for i = 1:n_order
        for g=1:n_repetitions
            %show prediction of amplitude
            scatter3(dirdiag_list(dr),order_list(i),regr_explv_pow_mar(g,dr,i),20,'b','filled');
            %show prediction of frequency
            %scatter3(dirdiag_list(dr),order_list(i),regr_explv_freq_mar(g,dr,i),20,'b','filled');
            hold on
        end
    end
end

title('predict amplitude HMM-MAR');
%title('predict frequency HMM-MAR');
xlabel('Dirdiag');
ylabel('order');
zlabel('CVEV');



%%

%%

%%

%% 

%% compute HMM-MAR sensitivity to amplitude and freq - vary nbr states
%% (SUPPLEMENTARY FIGURE 4)
%% 
n_trials = 40;
time_points_per_trial = 1000; % 4 secs
T = time_points_per_trial * ones(n_trials,1);  % length of data for each session
nchannels = 1; % regions or voxels

spont_options = struct();
spont_options.POW_AR_W = 0.995;
spont_options.FREQ_AR_W = 0.998;
spont_options.POW_RANGE = [0.1 5.0];
spont_options.FREQ_RANGE =[0.01 0.99]; 
spont_options.NOISE = 0.5;


n_repetitions = 10;
number_states = [3,5,7,9,11];

regrstate_explv_freq_mar = NaN(n_repetitions,length(number_states));
regrstate_explv_power_mar = NaN(n_repetitions,length(number_states));
regrstate_explv_freq_tde = NaN(n_repetitions,length(number_states));
regrstate_explv_power_tde = NaN(n_repetitions,length(number_states));

for g=1:n_repetitions
    % create signals
    % this gives us 40 trials of 4 secs each.
    X = zeros(time_points_per_trial, n_trials);
    Freq = zeros(time_points_per_trial, n_trials);
    Power = zeros(time_points_per_trial, n_trials);
    for i=1:n_trials
        [X(:,i),Freq(:,i),Power(:,i),spont_options,evoked_options] = simsignal(time_points_per_trial,1,spont_options,[],[],0);
    end


    %% run HMM-MAR 
    % have to reshape data as time*trial,channels
    data = reshape(X,[n_trials*time_points_per_trial,1]);

    % construct different HMM options
    template_configuration = struct();
    
    order = 3;

    % build config
    i = 1; 
    configurations = {}; % parameters of the HMM
    for k = number_states
        
        configurations{i} = template_configuration;
        configurations{i}.order = order;
        configurations{i}.K = k;
        configurations{i}.covtype = 'uniquediag';
        configurations{i}.useParallel=0;
        configurations{i}.DirichletDiag = 10000;
        configurations{i}.Fs = Fs;
        configurations{i}.fpass = [0 50];
        configurations{i}.p = 0;
        configurations{i}.to_do = [1 1];
        configurations{i}.Nf = 120;
        i = i+1;
       
    end

    L = length(configurations);

    Gamma_mar = cell(L,1);
    hmm_mar = cell(L,1);
    padded_gamma_mar = cell(L,1);
    spectra_mar = cell(1,L);
    for i = 1:L % for every config - every state number
        [hmm_mar{i},Gamma_mar{i}] = hmmmar(data,T,configurations{i});
        padded_gamma_mar{i} = padGamma(Gamma_mar{i},T,configurations{i});
        spectra_mar{i} = hmmspectramt(data,T,Gamma_mar{i},configurations{i});
        
        %% predict power and frequency for this repetition
        options = {};

        Ypboth = reshape(Power,[n_trials*time_points_per_trial,1]);
        [regrstate_explv_power_mar(g,i), beta_both_power] = ridge_regression_cv(padded_gamma_mar{i},Ypboth,T, options);

        Yf = reshape(Freq,[n_trials*time_points_per_trial,1]);
        [regrstate_explv_freq_mar(g,i), beta_both_freq] = ridge_regression_cv(padded_gamma_mar{i},Yf,T, options);
    end
    
   
    %% run HMM-TDE
    
    lags = 15;

    % build config
    i = 1; 
    configurations_tde = {}; % parameters of the HMM
    for k = number_states
        configurations_tde{i} = template_configuration;
        configurations_tde{i}.K = k;
        configurations_tde{i}.useParallel=0;
        configurations_tde{i}.DirichletDiag =10000;
        configurations_tde{i}.embeddedlags = [-lags(l):1:lags(l)];
        configurations_tde{i}.Fs = Fs;
        configurations_tde{i}.fpass = [0 50];
        configurations_tde{i}.p = 0;
        configurations_tde{i}.to_do = [1 1];
        configurations_tde{i}.Nf = 120;
    end

    Gamma_tde = cell(L,1);
    hmm_tde = cell(L,1);
    padded_gamma_tde = cell(L,1);
    spectra_tde = cell(L,1);
        


    for i = 1:L % for every config - every state number
        [hmm_tde{i},Gamma_tde{i}] = hmmmar(data,T,configurations_tde{i});
        padded_gamma_tde{i} = padGamma(Gamma_tde{i},T,configurations_tde{i});
        spectra_tde{i} = hmmspectramt(data,T,Gamma_tde{i},configurations_tde{i});
        
        %% predict power and frequency for this repetition
        options = {};

        Ypboth = reshape(Power,[n_trials*time_points_per_trial,1]);
        [regrstate_explv_power_tde(g,i), beta_both_power] = ridge_regression_cv(padded_gamma_tde{i},Ypboth,T, options);

        Yf = reshape(Freq,[n_trials*time_points_per_trial,1]);
        [regrstate_explv_freq_tde(g,i), beta_both_freq] = ridge_regression_cv(padded_gamma_tde{i},Yf,T, options);
    end
    
    
    
    
end


%% plot prediction results 
% !!!!! SHOW HMM-MAR or HMM-TDE

%m_freq = mean(regrstate_explv_freq_tde,1);
%m_pow = mean(regrstate_explv_power_tde,1);
%err_freq = std(regrstate_explv_freq_tde,1);
%err_pow = std(regrstate_explv_power_tde,1);

m_freq = mean(regrstate_explv_freq_mar,1);
m_pow = mean(regrstate_explv_power_mar,1);
err_freq = std(regrstate_explv_freq_mar,1);
err_pow = std(regrstate_explv_power_mar,1);


figure()
errorbar(m_freq,err_freq, 'LineWidth',2);
hold on
errorbar(m_pow,err_pow, 'Linewidth',2);
xlim([0.5 5.5])
xticks([1,2,3,4,5]);
xticklabels({'3','5','7','9','11'});
xlabel('nbr states');
ylabel('Explained variance');
legend({'frequency','amplitude'}, 'Location', 'west');
grid on
title('Predict frequency and amplitude from HMM-MAR states ')
%title('Predict frequency and amplitude from HMM-TDE states ')


%% plot states prob 
% 
plotting_range = [17*Fs:41*Fs];

plx = data - min(data);
plx = plx/max(plx);

% plot all 
for i=1:L
    figure();
    subplot(2,1,1);
    area(padded_gamma_mar{i}(plotting_range,:));
    ylabel('P(state)')
    hold on;
    plot(plx(plotting_range),'k');
    xticks([Fs:3*Fs:24*Fs]);
    xticklabels({'1', '4','7','10', '13','16','19','22'})
    xlim([0,13*Fs]);
    ylim([0 1]);
    xlabel('time (s)');
    title('HMM-MAR states probability', 'FontSize',14);
    
    subplot(2,1,2);
    area(padded_gamma_tde{i}(plotting_range,:));
    ylabel('P(state)')
    hold on;
    plot(plx(plotting_range),'k');
    xticks([Fs:3*Fs:14*Fs]);
    xticklabels({'1', '4','7','10', '13'})
    xlim([0,13*Fs]);
    ylim([0,1]);
    xlabel('time (s)');
    title('HMM-TDE states probability', 'FontSize',14);

end

%% compute and plot states power spectra 

% choose one of the analyses and the number of states
i = 2;
K = 3;
% in 
for k=1:K
    mar_state_pow_ch1(:,k) = spectra_mar{i}.state(k).psd(:,1,1);
    %mar_state_pow_ch2(:,i) = spectra_mar{1,1}.state(i).psd(:,2,2);
end


for i=1:K
    tde_state_pow_ch1(:,k) = spectra_tde{i}.state(k).psd(:,1,1);
    %tde_state_pow_ch2(:,i) = spectra_tde{1,1}.state(i).psd(:,2,2);
end



figure();
plot(mar_state_pow_ch1, 'Linewidth',2);
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
lgd = legend({'1', '2','3','4','5','6','7','8','9','10'});
lgd.Title.String = 'states';
lgd.FontSize = 14;
grid()
ylabel('power')
title('HMM-MAR frequency content of states', 'FontSize', 14);


figure();
plot(tde_state_pow_ch1, 'Linewidth',2);
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
ylabel('power')
%ylim([0 0.1])
grid()
lgd = legend({'1', '2','3','4','5','6','7','8','9','10'});
lgd.Title.String = 'states';
lgd.FontSize = 14;
title('HMM-TDE power content of states', 'FontSize',14);


