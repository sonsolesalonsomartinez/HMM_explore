
%% TEST HMM INFERENCE - TWO CHANNELS, SHOWING PERIODS OF COHERENCE

%% SCRIPT DESCRIPTION
% This script contains the code for testing HMM inference, on two-channel, 
% non stationary data exhibiting periods of between-channel coherence,
% producing results in Figure 5 and Supplementary Figure 5.

%% ANALYSIS 
% The two channels are generated as a combination of three independent
% signals, a,b, and c, as:
% x1 = rho(t) * a + (1-rho(t)) * c
% x2 = rho(t) * b + (1-rho(t)) * c
% so that when periods of maximal coherence is enforced (i.e., rho = 1)
% both channels are equal to signal c, when minimum coherence is
% imposed (rho = 0), channel 1 is equal to signal a and channel 2 
% is equal to signal b (critically, they are not necessarily uncorrelated!)

% Signals are generated as a long sequence of 50000 points and are then cut
% into 100 trials of 500 time points each for the regression analysis,
% predicting frequency or amplitude from the states time courses.

%% AUTHOR
% Laura Masaracchia 
% lauramacfin.au.dk
% Aarhus University
% June 2023

%% produce coherence coefficient rho as a smooth square wave

n_trials = 100;
time_points_per_trial = 500;

alpha =175; % the lower alpha, the slower oscillations. 
%The higher alpha, the faster oscillations of coherence
% t here made to be 1000 timepoints
t =linspace(0.1,175,n_trials*time_points_per_trial);

coh_coeff = (sin(t) + sin(3*t)/3 + 1 )/2 ;

%% plot rho
% figure();
% plot(t,coh_coeff);
% xlabel('t')
% ylabel('\rho')

%% set parameters for the analysis

n_repetitions = 2;
n_permutations = 2;
T = n_trials*time_points_per_trial;
Fs = 250;
K = 2;
% ranges to test
dirdiag_list = [1000,1000];%[100,1000,10000,100000,1000000,10000000, 100000000, 1000000000];
lags_list= [5,9];%[5,9,15,21,50]; % L 
steps_list = [1,1];%[1,1,3,3,10]; % S
order_list = [3,5];%[3,5,7,9]; % P
n_order = length(order_list);
n_lags = length(lags_list);
n_dir = length(dirdiag_list);

% signals parameters
spont_options = struct();
spont_options.POW_AR_W = 0.999;
spont_options.FREQ_AR_W = 0.999;
spont_options.POW_RANGE = [0.1 10.0];
spont_options.FREQ_RANGE = [0.01 0.99];
spont_options.NOISE = 1;
n_channels_in = 3;
nchannels = n_channels_in-1; 


regr_explv_coh_mar = NaN(n_repetitions,n_dir, n_order);
regr_explv_freq_mar = NaN(n_repetitions,nchannels, n_dir, n_order);
regr_explv_coh_tde = NaN(n_repetitions,n_dir, n_lags);
regr_explv_freq_tde = NaN(n_repetitions,nchannels, n_dir, n_lags);
switch_rate_mar = NaN(n_repetitions,n_dir);
switch_rate_tde = NaN(n_repetitions,n_dir);
regr_rand_explv_coh_tde = NaN(n_permutations,1);
regr_rand_explv_freq_tde = NaN(n_permutations,nchannels);


for dr=1:n_dir
    
    % %% example generate and plot signals
    % X_in = zeros(n_trials*time_points_per_trial, n_channels_in);
    % Freq_in = zeros(n_trials*time_points_per_trial, n_channels_in);
    % Power_in = zeros(n_trials*time_points_per_trial, n_channels_in);
    % for i=1:n_channels_in
    %     [X_in(:,i),Freq_in(:,i),Power_in(:,i),spont_options,evoked_options] = simsignal(n_trials*time_points_per_trial,1,spont_options,[],[],0);
    % end
    % 
    % X = get_coherent_signals(X_in, coh_coeff');
    % Freq = get_coherent_signals(Freq_in, coh_coeff');
    % Pow = get_coherent_signals(Power_in,coh_coeff');
    % coh_coeff_trials = reshape(coh_coeff',[time_points_per_trial, n_trials]);
    % Freq_trials = reshape(Freq, [time_points_per_trial,n_trials,nchannels]);
    % Pow_trials = reshape(Pow, [time_points_per_trial,n_trials,nchannels]);
    % 
    % plotting_range = 1:4000;
    % figure();
    % subplot(2,1,1)
    % plot(Freq(plotting_range,:));
    % xticks([250:500:4000]);
    % %xticklabels({'1', '4','7','10', '13', '16', '19'})
    % xticklabels({'', '','','', '', '', ''});
    % ylim([0 1.1]);
    % lgx = legend({'channel 1','channel 2'});
    % 
    % % title('signals')
    % subplot(2,1,2)
    % plot(coh_coeff(plotting_range));
    % xticks([250:500:4000]);
    % xticklabels({'1','3','5','7','9','11','13','15'})
    % xlabel('t (sec)')
    % title('coherence coefficient')
    % sgtitle('frequency');


    for g=1:n_repetitions
        disp(['repetition nbr' num2str(g)])
        
        % create independent signals
        X_in = zeros(n_trials*time_points_per_trial, n_channels_in);
        Freq_in = zeros(n_trials*time_points_per_trial, n_channels_in);
        Power_in = zeros(n_trials*time_points_per_trial, n_channels_in);
        for i=1:n_channels_in
            [X_in(:,i),Freq_in(:,i),Power_in(:,i),spont_options,evoked_options] = simsignal(n_trials*time_points_per_trial,1,spont_options,[],[],0);
        end
        
        % combine signals according to the coherence coefficient rho
        data = get_coherent_signals(X_in, coh_coeff');
        Freq = get_coherent_signals(Freq_in, coh_coeff');
        Pow = get_coherent_signals(Power_in,coh_coeff');
        coh_coeff_trials = reshape(coh_coeff',[time_points_per_trial, n_trials]);
        Freq_trials = reshape(Freq, [time_points_per_trial,n_trials,nchannels]);
        Pow_trials = reshape(Pow, [time_points_per_trial,n_trials,nchannels]);
        

        %% run HMM-MAR
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
            Ycoh = coh_coeff';
            options = {};
            [regr_explv_coh_mar(g,dr,o), beta_mar_pow] = ridge_regression_cv(padded_gamma_mar{o},Ycoh,T1, options);
            % For each channel regress frequency
            for ch=1:nchannels
                Yf = Freq(:,ch);
                [regr_explv_freq_mar(g,ch,dr,o), beta_mar_freq] = ridge_regression_cv(padded_gamma_mar{o},Yf,T1, options);
            end

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
            [regr_explv_coh_tde(g,dr,l), beta_tde_pow] = ridge_regression_cv(padded_gamma_tde{l},Ycoh,T1, options);
            % For each channel regress frequency
            for ch=1:nchannels
                Yf = Freq(:,ch);
                [regr_explv_freq_tde(g,ch,dr,l), beta_tde_freq] = ridge_regression_cv(padded_gamma_tde{l},Yf,T1, options);
            end
        end
            
       
    end

end

%% permute 10000 times, and store the error. 
for s=1:n_permutations    
    r = randperm(n_trials); 
    T1 = time_points_per_trial * ones(n_trials,1);
    % shuffle coh and regress
    random_coh = coh_coeff_trials(:,r);
    randomYcoh = reshape(random_coh,[n_trials*time_points_per_trial,1]);
    %[regr_rand_explv_coh_mar(g,s), beta_mar_coh_rand] = ridge_regression_cv(padded_gamma_mar{1,1},randomYcoh,T, options);
    [regr_rand_explv_coh_tde(s), beta_tde_coh_rand] = ridge_regression_cv(padded_gamma_tde{1},randomYcoh,T1, options);

    % shuffle freq and regress
    randomFreq = Freq_trials(:,r,:);
    randomYf = reshape(randomFreq,[n_trials*time_points_per_trial,nchannels]);

    for ch=1:nchannels
        %[regr_rand_explv_freq_mar(g,s,ch), rand_beta_freq_mar] = ridge_regression_cv(padded_gamma_mar{1,1},randomYf(:,ch),T, options);
        [regr_rand_explv_freq_tde(s,ch), rand_beta_freq_tde] = ridge_regression_cv(padded_gamma_tde{1},randomYf(:,ch),T1, options);
    end
end

%%
%% compute empirical correlation
% 
% emp_coh = NaN(T,2);
% emp_corr = NaN(T,1);
% half_wind_size = 50;
% for sw=1:T
%     if sw<=half_wind_size
%         window_data = data(1:sw+half_wind_size,:);
%     elseif sw>T-half_wind_size
%         window_data = data(sw-half_wind_size:T,:);
%     else
%         window_data = data(sw-half_wind_size:sw+half_wind_size,:);
%     end
%     inst_corr_mat = corrcoef(window_data);
%     [emp_coh_f,f] = mscohere(window_data(:,1),window_data(:,2),[],[],[],250);
%     %inst_corr_mat = corrcov(window_data);
%     emp_corr(sw) = inst_corr_mat(2,1);
%     [emp_coh(sw,1), emp_coh(sw,2)] = max(emp_coh_f);
%     %emp_ps(sw) = sum(window_data(:,1).* window_data(:,2));
%     
% end

%% plot states probability time courses 
Fs = 250;
plotting_range = [1*Fs:15*Fs];

plx = data - min(data);
plx = plx/max(plx);


figure();
subplot(2,1,1);
area(padded_gamma_mar{1}(plotting_range,:));
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
area(padded_gamma_tde{1}(plotting_range,:));
ylabel('P(state)')
hold on;
plot(plx(plotting_range),'k');
xticks([Fs:3*Fs:14*Fs]);
xticklabels({'1', '4','7','10', '13'})
xlim([0,13*Fs]);
ylim([0,1]);
xlabel('time (s)');
title('HMM-TDE states probability', 'FontSize',14);


%% compute power and coherence of states
%plot analysis 1
for i=1:K
    mar_state_coh(:,i) = spectra_mar{1}.state(i).coh(:,2,1);
    mar_state_pow_ch1(:,i) = spectra_mar{1}.state(i).psd(:,1,1);
    mar_state_pow_ch2(:,i) = spectra_mar{1}.state(i).psd(:,2,2);
end


for i=1:K
    tde_state_coh(:,i) = spectra_tde{1}.state(i).coh(:,2,1);
    tde_state_pow_ch1(:,i) = spectra_tde{1}.state(i).psd(:,1,1);
    tde_state_pow_ch2(:,i) = spectra_tde{1}.state(i).psd(:,2,2);
end


%% plot power and coherence of states
figure();
plot(mar_state_coh, 'Linewidth',2);
lgx = legend({'state 1', 'state 2', 'state 3', 'state 4'});
lgx.FontSize = 16;
xticks([0:20:180]);
xticklabels({'0','5','10','15','20','25','30','35','40','45'});
xlabel('frequency (Hz)');
grid()
ylabel('coherence');
title('HMM-MAR states coherence', 'FontSize',14)

figure();
subplot(1,2,1)
plot(mar_state_pow_ch1, 'Linewidth',2);
title('channel 1');
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
%ylim([0 0.08])
grid()
ylabel('power')
subplot(1,2,2)
plot(mar_state_pow_ch2, 'Linewidth',2);
title('channel 2')
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
grid()
lgx = legend({'state 1','state 2','state 3', 'state 4'});
lgx.FontSize = 14;
ylabel('power')
sgtitle('HMM-MAR frequency content of states', 'FontSize',16);


figure();
plot(tde_state_coh, 'Linewidth',2);
lgx = legend({'state 1', 'state 2', 'state 3', 'state 4'});
lgx.FontSize = 16;
xticks([0:20:180]);
xticklabels({'0','5','10','15','20','25','30','35','40','45'});
xlabel('frequency (Hz)');
ylabel('coherence');
grid()
title('HMM-TDE states coherence', 'FontSize',14)


figure();
subplot(1,2,1)
plot(tde_state_pow_ch1, 'Linewidth',2);
title('channel 1');
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
ylabel('power')
grid()
subplot(1,2,2)
plot(tde_state_pow_ch2, 'Linewidth',2);
title('channel 2')
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
ylabel('power')
grid()
lgx = legend({'state 1','state 2','state 3', 'state 4'});
lgx.FontSize = 14;
sgtitle('HMM-TDE frequency content of states', 'FontSize',16);


%% 3D plot of prediction accuracy
%as function of switching rate and hyperparameters 
% !!!! PLOT EITHER ACCURACY ON COHERENCE OR ON FREQUENCY !!!!!!!!

%show prediction on coherence
%Z_MAR = squeeze(mean(regr_explv_coh_mar,1));
%Z_TDE = squeeze(mean(regr_explv_coh_tde,1));

%show prediction on frequency
% also average across channels
Z_TDE = squeeze(mean(squeeze(mean(regr_explv_freq_tde,1)),1));
Z_MAR = squeeze(mean(squeeze(mean(regr_explv_freq_mar,1)),1));


[X1,Y1] = meshgrid(lags_list, dirdiag_list);

figure()
surf(X1,Y1,Z_TDE);
colormap autumn

xlabel('lags')
ylabel('\delta')
zlabel('CVEV')
%title('predict coherence HMM-TDE')
title('predict frequency HMM-TDE')


[X2,Y2] = meshgrid(order_list, dirdiag_list);

figure()
surf(X2,Y2,Z_MAR);
colormap winter
xlabel('order')
ylabel('\delta')
zlabel('CVEV')
%title('predict coherence HMM-MAR')
title('predict frequency HMM-TDE')

%% 2D PLOT of prediction accuracy 
%as a function of switching rate
% !!!!!!! SHOW EITHER PREDICTION ON FREQUENCY OR ON AMPLITUDE !!!!!!!!

% show prediction on amplitude
%y_TDE = squeeze(regr_explv_coh_tde(:,dr,i))';
%y_MAR = squeeze(regr_explv_coh_tde(:,dr,i))';

%show prediction on frequency
y_TDE = squeeze(mean(regr_explv_freq_tde(:,:,dr,i),1))';
y_MAR = squeeze(mean(regr_explv_freq_tde(:,:,dr,i),1))';

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

%title('predict coherence from states');
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


