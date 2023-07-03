
%% TEST HMM GENERALIZATION - LFP DATA

%% SCRIPT DESCRIPTION
% This script contains the code for testing HMM generalization to LFP data,
% producing results in Figure 6.

%% AUTHOR
% Laura Masaracchia 
% lauramacfin.au.dk
% Aarhus University
% June 2023

%% Running HMM on LFP data
% PLEASE NOTE THE DATA ARE NOT PUBLISHED YET
% load LFP data
load("hippo_10min_250hz.mat");

% get only 2 channels
chosen_channels = [4,8];
nchannels = length(chosen_channels);

% take only 50000 points - like for all the other analyses
T = 50000;
X = X_hip_10min(1:T,chosen_channels);
Fs = 250;

%% plot data
figure();
plot(X(1:3000,:));
xticks([250:500:3000])
xticklabels({'1','3','5','7','9','11','13','15'})
xlim([0 3000])
xlabel('t (sec)');
lgx = legend({'channel 1', 'channel 2'});
lgx.FontSize = 16;
title('hippocampus LFP, 2 channels', 'FontSize', 16)

%% freq content of data
figure();
for j=1:nchannels
    subplot(2,2,j)
    [power_dist_data, f_data] = pwelch(X(:,j),[],[],[],Fs);
    % plot them together
    plot(f_data,power_dist_data, 'k');
    title(['channel ' num2str(j)]);
    xlim([-0.1 50])
    xlabel('frequency (Hz)');
    ylabel('power')

end

sgtitle('frequency content of data')


%% compute correlations

% emp_corr = NaN(T,6);
% half_wind_size = 100;
% for sw=1:T
%     if sw<=half_wind_size
%         window_data = X(1:sw+half_wind_size,:);
%     elseif sw>T-half_wind_size
%         window_data = X(sw-half_wind_size:T,:);
%     else
%         window_data = X(sw-half_wind_size:sw+half_wind_size,:);
%     end
%     inst_corr_mat = corrcoef(window_data);
%     %inst_corr_mat = corrcov(window_data);
%     emp_corr(sw,1) = inst_corr_mat(2,1);
%     emp_corr(sw,2) = inst_corr_mat(3,2);
%     emp_corr(sw,3) = inst_corr_mat(4,3);
%     emp_corr(sw,3) = inst_corr_mat(3,1);
%     emp_corr(sw,5) = inst_corr_mat(4,2);
%     emp_corr(sw,6) = inst_corr_mat(4,1);
% end

emp_coh = NaN(T,2);
emp_corr = NaN(T,1);
half_wind_size = 50;
for sw=1:T
    if sw<=half_wind_size
        window_data = X(1:sw+half_wind_size,:);
    elseif sw>T-half_wind_size
        window_data = X(sw-half_wind_size:T,:);
    else
        window_data = X(sw-half_wind_size:sw+half_wind_size,:);
    end
    inst_corr_mat = corrcoef(window_data);
    [emp_coh_f,f] = mscohere(window_data(:,1),window_data(:,2),[],[],[],250);
    %inst_corr_mat = corrcov(window_data);
    emp_corr(sw) = inst_corr_mat(2,1);
    [emp_coh(sw,1), emp_coh(sw,2)] = max(emp_coh_f);
    %emp_ps(sw) = sum(window_data(:,1).* window_data(:,2));
    
end



%% plot correlation
figure();
plot(emp_corr(1:3000), 'Linewidth', 2);
hold on;
%plot(emp_coh(1:5000,1),'--', 'Linewidth', 2)
hold on
plot(smoothdata(emp_coh(1:3000,1), 'gaussian',50), '-.', 'Linewidth', 2);
legend({'corr', 'coh' }, 'Location', 'SouthWest');
%ylim([0 1.1])
title('correlation and coherence of signals - ws 500msec')
xticks([250:500:3000])
xticklabels({'1','3','5','7','9','11','13','15'})
xlim([0 3000])
grid();
xlabel('t (sec)')
ylabel('abs(corr)')


%% define nbr states
%number_states = [3,5];
number_states = [3];%
%smoothed_emp_coh = smoothdata(emp_coh(:,1), 'gaussian',20);
%% run HMM-MAR
% build different config to test
config_temp_mar = struct();
orders = [5];

% build config
i = 1; 
configurations_mar = {}; % parameters of the HMM
for k = number_states
    for order = orders
        configurations_mar{i} = config_temp_mar;
        configurations_mar{i}.order = order;
        configurations_mar{i}.K = k;
        configurations_mar{i}.covtype = 'uniquediag';
        configurations_mar{i}.useParallel=0;
        configurations_mar{i}.DirichletDiag = 1000;
        i = i + 1;
    end
end

L = length(configurations_mar);

Gamma_mar = cell(L,1);
hmm_mar = cell(L,1);

padded_gamma_mar = cell(L,1);
for i = 1:L
    [hmm_mar{i},Gamma_mar{i}] = hmmmar(X,T,configurations_mar{i});
    padded_gamma_mar{i} = padGamma(Gamma_mar{i},T,configurations_mar{i});
end

%save('gamma_mar_LFP_o5_dirdiag5k_2ch.mat','hmm_mar','Gamma_mar', 'padded_gamma_mar', 'T');

%% run HMM-TDE
config_temp_tde = struct();
lags = [3];

% build config
i = 1; 
configurations_tde = {}; % parameters of the HMM
for k = number_states
    for l= lags
        configurations_tde{i} = config_temp_tde;
        configurations_tde{i}.order = 0;
        configurations_tde{i}.zeromean = 1;
        configurations_tde{i}.K = k;
        configurations_tde{i}.inittype = 'HMM-MAR';
        configurations_tde{i}.covtype = 'full';
        configurations_tde{i}.useParallel=0;
        configurations_tde{i}.DirichletDiag = 1000;
        configurations_tde{i}.embeddedlags = [-l:l];
        i = i + 1;
    end
end

L = length(configurations_tde);
Gamma_tde = cell(L,1);
hmm_tde = cell(L,1);
padded_gamma_tde = cell(L,1);

for i = 1:L
    [hmm_tde{i},Gamma_tde{i}] = hmmmar(X,T,configurations_tde{i});
    padded_gamma_tde{i} = padGamma(Gamma_tde{i},T,configurations_tde{i});

end

%save('gamma_tde_LFP_lag15_dirdiag10k_2ch.mat','hmm_tde','Gamma_tde', 'padded_gamma_tde', 'T');


%% PLOT states time courses
plx = X - min(X);
plx = plx./max(plx);

plotting_range = [8001:10000];

for i=1:length(number_states)
    figure()
    subplot(2,1,1);
    area(padded_gamma_mar{i}(plotting_range,:));
    hold on;
    plot(plx(plotting_range,:),'k');
    xticks([250:500:2000]);
    xticklabels({});
    ylim([0 1]);
    title('HMM-MAR - order 7 - dir diag 5k')
    subplot(2,1,2);
    area(padded_gamma_tde{i}(plotting_range,:));
    hold on;
    plot(plx(plotting_range,:),'k');
    xticks([250:500:2000]);
    xticklabels({'1','3', '5','7','9','11','13','15','17','19','21','23'});
    ylim([0 1]);
    title('HMM-TDE - lag 15 - dir diag 5k');
%     subplot(3,1,3);
%     plot(smoothed_emp_coh(plotting_range,1),'k', 'LineWidth',2);
%     grid();
%     xticks([250:500:2000]);
%     xticklabels({'1','3', '5','7','9','11','13','15','17','19','21','23'});
%     xlabel('t (sec)');
%     %ylabel('coh')
%     title('signal coherence - ws 500msec')
%     %legend({'1 & 2','2 & 3','3 & 4','1 & 3','2 & 4','1 & 4'}, 'Orientation', 'Horizontal', 'Location', 'North');
%     ylim([0 1.1])
    sgtitle(['HMMs state time courses comparison - k = ' num2str(number_states(i))]);
    
end

    
%%   spectra analyses
    
spectra_mar = cell(L,1);
spectra_tde = cell(L,1);
%maxFO_mar = {};
%FO_mar = {};
%LifeTimes_mar = {};
%Intervals_mar = {};
%SwitchingRate_mar = {};

% FO, life intervals and SR
for i=1:L
    configurations_mar{i}.Fs = Fs; % Sampling rate 
    configurations_mar{i}.fpass = [1 45];  % band of frequency you're interested in
    configurations_mar{i}.p = 0; % interval of confidence  
    configurations_mar{i}.to_do = [1 1];
    configurations_mar{i}.Nf = 120; 

    configurations_tde{i}.Fs = Fs;
    configurations_tde{i}.fpass = [1 45];
    configurations_tde{i}.p = 0;
    configurations_tde{i}.to_do = [1 1];
    configurations_tde{i}.Nf = 120;

    spectra_mar{i} = hmmspectramt(X,T,Gamma_mar{i},configurations_mar{i});
    spectra_tde{i} = hmmspectramt(X,T,Gamma_tde{i},configurations_tde{i});

    % Some useful information about the dynamics
    %maxFO_mar{i} = getMaxFractionalOccupancy(Gamma_mar{i},T,configurations_mar{i}); % useful to diagnose if the HMM 
            % is capturing dynamics or grand between-subject 
            % differences (see Wiki)
    %FO_mar{i} = getFractionalOccupancy (Gamma_mar{i},T,configurations_mar{i}); % state fractional occupancies per session
    %LifeTimes_mar{i} = getStateLifeTimes (Gamma_mar{i},T,configurations_mar{i}); % state life times
    %Intervals_mar{i} = getStateIntervalTimes (Gamma_mar{i},T,configurations_mar{i}); % interval times between state visits
    %SwitchingRate_mar{i} =  getSwitchingRate(Gamma_mar{i},T,configurations_mar{i}); % rate of switching between stats
end


%% compute power and coherence of states


for i=1:3
    mar_state_coh(:,i) = spectra_mar{1}.state(i).coh(:,2,1);
    mar_state_pow_ch1(:,i) = spectra_mar{1}.state(i).psd(:,1,1);
    mar_state_pow_ch2(:,i) = spectra_mar{1}.state(i).psd(:,2,2);
end


for i=1:3
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
ylim([0 0.8])
title('HMM-MAR states coherence', 'FontSize',14)
%%
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
%ylim([0 0.08])
ylabel('power')
sgtitle('HMM-MAR frequency content of states', 'FontSize',16);



%%

figure();
plot(tde_state_coh, 'Linewidth',2);
lgx = legend({'state 1', 'state 2', 'state 3', 'state 4'});
lgx.FontSize = 16;
xticks([0:20:180]);
xticklabels({'0','5','10','15','20','25','30','35','40','45'});
xlabel('frequency (Hz)');
ylabel('coherence');
grid()
ylim([0 0.8]);
title('HMM-TDE states coherence', 'FontSize',14)
%%

figure();
subplot(1,2,1)
plot(tde_state_pow_ch1, 'Linewidth',2);
title('channel 1');
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
%xticklabels({'0','5','10','15','20','25','30','35','40','45'});
xlabel('frequency (Hz)');
ylabel('power')
grid()
%ylim([0 0.08])
subplot(1,2,2)
plot(tde_state_pow_ch2, 'Linewidth',2);
title('channel 2')
xticks([0:40:180]);
xticklabels({'0','10','20','30','40'});
xlabel('frequency (Hz)');
ylabel('power')
%ylim([0 0.08])
grid()
lgx = legend({'state 1','state 2','state 3', 'state 4'});
lgx.FontSize = 14;
sgtitle('HMM-TDE frequency content of states', 'FontSize',16);


%%
%%
%%
%%
%% other states differences


maxFO_hmmmar = {};
FO_hmmmar = {};
LifeTimes_hmmmar = {};
Intervals_hmmmar = {};
SwitchingRate_hmmmar = {};

% FO, life intervals and SR
for i=1:length(configurations)
    spectra_hmmmar{i} = hmmspectramar(X,T,[],Gamma{i},options_spectra);

    % Some useful information about the dynamics
    maxFO_hmmmar{i} = getMaxFractionalOccupancy(Gamma{i},T,configurations{i}); % useful to diagnose if the HMM 
            % is capturing dynamics or grand between-subject 
            % differences (see Wiki)
    FO_hmmmar{i} = getFractionalOccupancy (Gamma{i},T,configurations{i}); % state fractional occupancies per session
    LifeTimes_hmmmar{i} = getStateLifeTimes (Gamma{i},T,configurations{i}); % state life times
    Intervals_hmmmar{i} = getStateIntervalTimes (Gamma{i},T,configurations{i}); % interval times between state visits
    SwitchingRate_hmmmar{i} =  getSwitchingRate(Gamma{i},T,configurations{i}); % rate of switching between stats
end


%% check similarities of different gammas given different HMM configurations

[sim_matrix_3_states,a,b] = getGammaSimilarity(padded_gamma_mar{1},padded_gamma_tde{1});



%% check gammas gradient


gradient_gamma_mar_3states = mean(sum(abs(padded_gamma_mar{1}(1:end-1,:) -  padded_gamma_mar{1}(2:end,:))));

gradient_gamma_tde_3states = mean(sum(abs(padded_gamma_tde{1}(1:end-1,:) -  padded_gamma_tde{1}(2:end,:))));
%% check max prob of occupancy per time point (across state), produce histogram


max_mar_state_prob_timepoint = max(padded_gamma_mar{1},[],2);

max_tde_state_prob_timepoint = max(padded_gamma_tde{1},[],2);

%%
figure()
subplot(1,2,1)
histogram(max_tde_state_prob_timepoint);
title('tde max prob')

subplot(1,2,2)
histogram(max_mar_state_prob_timepoint);
title('mar max prob')

