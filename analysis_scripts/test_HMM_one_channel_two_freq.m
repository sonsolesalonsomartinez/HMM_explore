%% TEST HMM INFERENCE - ONE CHANNEL, TWO CONCURRING FREQUENCIES

%% SCRIPT DESCRIPTION
% This script contains the code for testing HMM inference, on one-channel, 
% non stationary data with two concurring frequencies per time point,
% producing results in Figure 4.

%% ANALYSIS 

% generate fast signal and slow signal, then combine them as
% signal= alpha * fast_signal + (1-alpha) * slow_signal
% and examine how the HMM works varying alpha from 0.01 to 0.5
% show how well the HMM characterizes changes in frequency (i.e. predict
% frequency from states) as a function of alpha

% Signals are generated as a long sequence of 50000 points and are then cut
% into 100 trials of 500 time points each for the regression analysis,
% predicting frequency or amplitude from the states time courses.
%% AUTHOR
% Laura Masaracchia 
% lauramacfin.au.dk
% Aarhus University
% June 2023


%% experiment repetitions
n_repetitions = 10;

%%
n_trials = 100;
time_points_per_trial = 500;

%% fast-low power signal options
spont_options_fast = struct();
spont_options_fast.POW_AR_W = 0.999;
spont_options_fast.FREQ_AR_W = 0.995;
spont_options_fast.POW_RANGE = [1.0 10.0];
spont_options_fast.FREQ_RANGE =[0.4 0.99]; 
spont_options_fast.NOISE =1;

%% slow-high power signal options
spont_options_slow = struct();
spont_options_slow.POW_AR_W = 0.999;
spont_options_slow.FREQ_AR_W = 0.995;
spont_options_slow.POW_RANGE = [1.0 10.0];
spont_options_slow.FREQ_RANGE =[0.01 0.4]; 
spont_options_slow.NOISE =1;

%% alpha
alpha_list = [0.2,0.3,0.4,0.5];
n_alpha = length(alpha_list);
states_number_list = [2,3,5];
n_states = length(states_number_list);

%% initialization
regr_explv_freq_fast_mar = NaN(n_repetitions, n_alpha, n_states);
regr_explv_pow_fast_mar = NaN(n_repetitions, n_alpha, n_states);
regr_explv_freq_fast_tde = NaN(n_repetitions, n_alpha, n_states);
regr_explv_pow_fast_tde = NaN(n_repetitions, n_alpha, n_states);
regr_explv_freq_slow_mar = NaN(n_repetitions, n_alpha, n_states);
regr_explv_pow_slow_mar = NaN(n_repetitions, n_alpha, n_states);
regr_explv_freq_slow_tde = NaN(n_repetitions, n_alpha, n_states);
regr_explv_pow_slow_tde = NaN(n_repetitions, n_alpha, n_states);


%% plotting range
% multiples of Fs
Fs = 250;
ll = 1;
ul = 6;
st = 2;

plotting_range = [ll*Fs:ul*Fs];
          
%% start analysis
for k=1:n_states
    % initialize analysis parameters
    states_number = states_number_list(k);
    order = 7;
    lag = 15;
    dird = 1000000;

    % HMM-MAR configs
    configurations_mar = struct();
    configurations_mar.order = order;
    configurations_mar.K = states_number;
    configurations_mar.covtype = 'uniquediag';
    configurations_mar.useParallel=0;
    configurations_mar.DirichletDiag = dird;
    configurations_mar.Fs = Fs; % Sampling rate 
    configurations_mar.standardise = 0; % Sampling rate 
    configurations_mar.fpass = [0 50];  % band of frequency you're interested in
    configurations_mar.p = 0; % interval of confidence  
    configurations_mar.to_do = [1 1];
    configurations_mar.Nf = 120; 

    Gamma_mar = cell(n_alpha,1);
    hmm_mar = cell(n_alpha,1);
    padded_gamma_mar = cell(n_alpha,1);
    spectra_mar = cell(n_alpha,1);

    % HMM-TDE configs
    configurations_tde = struct();
    configurations_tde.K = states_number;
    configurations_tde.useParallel= 0;
    configurations_tde.DirichletDiag = dird;
    configurations_tde.embeddedlags = [-lag:lag];
    configurations_tde.Fs = Fs;
    configurations_tde.fpass = [0 50];
    configurations_tde.p = 0;
    configurations_tde.to_do = [1 1];
    configurations_tde.Nf = 120;

    Gamma_tde = cell(n_alpha,1);
    hmm_tde = cell(n_alpha,1);
    padded_gamma_tde = cell(n_alpha,1);
    spectra_tde = cell(n_alpha,1);

    for a = 1:n_alpha
        alpha = alpha_list(a);    

        for g=1:n_repetitions
            % create signals
            % fast signal
            [X_fast,Freq_fast,Power_fast,spont_options_fast,evoked_options] = simsignal(n_trials*time_points_per_trial,1,spont_options_fast,[],[],0);
            % slow signal
            [X_slow,Freq_slow,Power_slow,spont_options_slow,evoked_options] = simsignal(n_trials*time_points_per_trial,1,spont_options_slow,[],[],0);
            % combine
            X = alpha * X_fast + (1-alpha) * X_slow;
            Freq = alpha * Freq_fast + (1-alpha) * Freq_slow;
            Power = alpha * Power_fast + (1-alpha) * Power_slow;
            
            % compute signal power spectrum
            options = struct();
            options.Fs = Fs;
            options.fpass = [0 50];
            options.Nf = 120;
            spectra_global = hmmspectramt(X,n_trials*time_points_per_trial,[], options);

            %% plot signal example
%     
%             figure();
%             %subplot(3,1,1);
%             
%             plot(X(plotting_range), 'k');
%             xticks([ll*Fs:st*Fs:ul*Fs]);
%             %xticklabels({'1', '3','5','7', '9', '11','13'})
%             %xticklabels({'', '','','', '', '', ''});
%             xlim([0,(ul-ll)*Fs]);
%     %         subplot(3,1,2);
%     %         plot(Freq(plotting_range), 'k', 'LineWidth', 2);
%     %         xticks([ll*Fs:st*Fs:ul*Fs]);
%     %         xlim([0,(ul-ll)*Fs]);
%     %         yticks([0 0.5 1.0]);
%     %         ylabel('inst freq (Hz)');
%     %         yticklabels({'0','25', '50'});
%     %         %ylim([0.09 0.12]);
%     %         %xticklabels({'1', '4','7','10', '13', '16', '19'})
%     %         xticklabels({''});
%     %         subplot(3,1,3);
%     %         plot(Power(plotting_range), 'k', 'LineWidth',2);
%             xlabel('time (s)');
%             xlim([0,(ul-ll)*Fs]);
%             xticks([ll*Fs:st*Fs:ul*Fs]);
%             xticklabels({'1', '3','5','7','9','11', '13'})
%             ylabel('amplitude');
%     
%             sgtitle(['example of signal , alpha = ', num2str(alpha)]);

            %% run HMM-MAR 

            [hmm_mar{a},Gamma_mar{a}] = hmmmar(X,n_trials*time_points_per_trial,configurations_mar);
            padded_gamma_mar{a} = padGamma(Gamma_mar{a},n_trials*time_points_per_trial,configurations_mar);
            spectra_mar{a} = hmmspectramt(X,n_trials*time_points_per_trial,Gamma_mar{a},configurations_mar);

            for i=1:states_number
                mar_state_ps(:,i) = spectra_mar{a}.state(i).psd;
            end


            %% run HMM-TDE

            [hmm_tde{a},Gamma_tde{a}] = hmmmar(X,n_trials*time_points_per_trial,configurations_tde);
            padded_gamma_tde{a} = padGamma(Gamma_tde{a},n_trials*time_points_per_trial,configurations_tde);
            spectra_tde{a} = hmmspectramt(X,n_trials*time_points_per_trial,Gamma_tde{a},configurations_tde);

            for i=1:states_number
                tde_state_ps(:,i) = spectra_tde{a}.state(i).psd;
            end

            state_f = spectra_tde{a}.state(1).f;

            %% plot HMM state time courses
%           
%             plx = X - min(X);
%             plx = plx/max(plx);         
%             figure();
%             subplot(2,1,1);
%             area(padded_gamma_mar{a}(plotting_range,:));
%             ylabel('P(state)')
%             hold on;
%             plot(plx(plotting_range),'k');
%             xticks([ll*Fs:st*Fs:ul*Fs]);
%             xticklabels({'1', '3','5','7','9', '11','13'})
%             xlim([0,(ul-ll)*Fs]);
%             ylim([0 1]);
%             xlabel('time (s)');
%             title('HMM-MAR ');
%             subplot(2,1,2);
%             area(padded_gamma_tde{a}(plotting_range,:));
%             ylabel('P(state)')
%             hold on;
%             plot(plx(plotting_range),'k');
%             xticks([ll*Fs:st*Fs:ul*Fs]);
%             xticklabels({'1', '3','5','7','9','11', '13'})
%             xlim([0,(ul-ll)*Fs]);
%             ylim([0,1]);
%             xlabel('time (s)');
%             title('HMM-TDE ');
%             sgtitle(['HMM states time courses, alpha = ',num2str(alpha)])

            %% plot HMM states spectra 
%             
%             figure();
%             subplot(3,1,1)
%             plot(spectra_global.state.f, spectra_global.state.psd, 'k')
%             xlabel('frequency (Hz)');
%             ylabel('power');
%             title('signal')
%             
%             subplot(3,1,2)
%             for i=1:k
%                 plot(state_f, mar_state_ps(:,i));
%                 hold on
%             end
%             xlabel('frequency (Hz)');
%             ylabel('power');
%             title('HMM-MAR')
%             subplot(3,1,3)
%             for i=1:k
%                 plot(state_f, tde_state_ps(:,i));
%                 hold on
%             end
%             xlabel('frequency (Hz)');
%             ylabel('power');
%             title('HMM-TDE')
%             sgtitle(['states and signal ps, alpha = ',num2str(alpha)])
         
            

            %% predict frequency and power for this repetition
            options = {};
            T = time_points_per_trial * ones(n_trials,1); 

            [regr_explv_freq_fast_mar(g, a, k), beta_mar_freq] = ridge_regression_cv(padded_gamma_mar{a},Freq_fast,T, options);
            [regr_explv_freq_fast_tde(g, a, k), beta_tde_pow] = ridge_regression_cv(padded_gamma_tde{a},Freq_fast,T, options);
            [regr_explv_pow_fast_mar(g, a, k), beta_mar_freq] = ridge_regression_cv(padded_gamma_mar{a},Power_fast,T, options);
            [regr_explv_pow_fast_tde(g, a, k), beta_tde_pow] = ridge_regression_cv(padded_gamma_tde{a},Power_fast,T, options);
            [regr_explv_freq_slow_mar(g, a, k), beta_mar_freq] = ridge_regression_cv(padded_gamma_mar{a},Freq_slow,T, options);
            [regr_explv_freq_slow_tde(g, a, k), beta_tde_pow] = ridge_regression_cv(padded_gamma_tde{a},Freq_slow,T, options);
            [regr_explv_pow_slow_mar(g, a, k), beta_mar_freq] = ridge_regression_cv(padded_gamma_mar{a},Power_slow,T, options);
            [regr_explv_pow_slow_tde(g, a, k), beta_tde_pow] = ridge_regression_cv(padded_gamma_tde{a},Power_slow,T, options);

        end

    end
end

%% save prediction

save('mar_predict_freq_power_multiple_freq.mat','regr_explv_freq_slow_mar', 'regr_explv_freq_fast_mar', 'regr_explv_pow_slow_mar', 'regr_explv_pow_fast_mar');
save('tde_predict_freq_power_multiple_freq.mat','regr_explv_freq_slow_tde', 'regr_explv_freq_fast_tde', 'regr_explv_pow_slow_tde', 'regr_explv_pow_fast_tde');


%% average explained variance as a function of alpha

p_fast_tde = squeeze(mean(regr_explv_pow_fast_tde,1));
err_p_fast_tde = squeeze(std(regr_explv_pow_fast_tde,1));
p_slow_tde = squeeze(mean(regr_explv_pow_slow_tde,1));
err_p_slow_tde = squeeze(std(regr_explv_pow_slow_tde,1));
f_slow_tde = squeeze(mean(regr_explv_freq_slow_tde,1));
err_f_slow_tde = squeeze(std(regr_explv_freq_slow_tde,1));
f_fast_tde = squeeze(mean(regr_explv_freq_fast_tde,1));
err_f_fast_tde = squeeze(std(regr_explv_freq_fast_tde,1));

p_fast_mar = squeeze(mean(regr_explv_pow_fast_mar,1));
err_p_fast_mar = squeeze(std(regr_explv_pow_fast_mar,1));
p_slow_mar = squeeze(mean(regr_explv_pow_slow_mar,1));
err_p_slow_mar = squeeze(std(regr_explv_pow_slow_mar,1));
f_slow_mar = squeeze(mean(regr_explv_freq_slow_mar,1));
err_f_slow_mar = squeeze(std(regr_explv_freq_slow_mar,1));
f_fast_mar = squeeze(mean(regr_explv_freq_fast_mar,1));
err_f_fast_mar = squeeze(std(regr_explv_freq_fast_mar,1));



%% plot results
% colors 
% blue	"#0072BD"
% light blue "#4DBEEE"
% orange "#D95319"
% red "#A2142F"


for a = 1:n_alpha
    figure();
    subplot(2,1,1)
    errorbar(p_fast_mar(a,:),err_p_fast_mar(a,:), 'LineWidth',2, 'color', "#D95319");
    hold on
    errorbar(p_slow_mar(a,:),err_p_slow_mar(a,:), 'LineWidth',2, 'color', "#A2142F");
    hold on
    errorbar(f_fast_mar(a,:),err_f_fast_mar(a,:), 'LineWidth',2, 'color', "#4DBEEE");
    hold on
    errorbar(f_slow_mar(a,:),err_f_slow_mar(a,:), 'LineWidth',2, 'color', "#0072BD");
    hold on
    title('HMM-MAR')
    xlim([0.3,3.7])
    ylim([-0.5 1])
    xticks([1,2,3]);
    grid on
    xticklabels({'2','3','5'});
    xlabel('states number');
    ylabel('Explained variance');
    legend({'a fast','a slow', 'f fast','f slow'}, 'Location', 'northwest');
    subplot(2,1,2)
    errorbar(p_fast_tde(a,:),err_p_fast_tde(a,:), 'LineWidth',2, 'color', "#D95319");
    hold on
    errorbar(p_slow_tde(a,:),err_p_slow_tde(a,:), 'LineWidth',2, 'color', "#A2142F");
    hold on
    errorbar(f_fast_tde(a,:),err_f_fast_tde(a,:), 'LineWidth',2, 'color', "#4DBEEE");
    hold on
    errorbar(f_slow_tde(a,:),err_f_slow_tde(a,:), 'LineWidth',2, 'color', "#0072BD");
    hold on
    xticks([1,2,3]);
    xticklabels({'2','3','5'});
    xlim([0.3,3.7])
    ylim([-0.5 1])
    title('HMM-TDE');
    xlabel('states number');
    grid on
    ylabel('Explained variance');
    legend({'a fast','a slow', 'f fast','f slow'}, 'Location', 'northwest');
    sgtitle(['alpha = ', num2str(alpha_list(a))])

end

