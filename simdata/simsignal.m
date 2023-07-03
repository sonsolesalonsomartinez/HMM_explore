function [X,Freq,Power,spont_options,evoked_options] = ....
    simsignal(T,nchan,spont_options,evoked_options,head_model,show)
% Generates synthetic "ephys" data, with spontaneously time-varying  
% power and frequency, and various options to introduce different types of 
% stimulus-evoked effects, for nchan channels (2nd argument), and time
% series which lengths are specified by T (1st argument; each value of T
% would correspond to a scanning session) - data are sampled at freq = 6Hz.
% 
% The spontaneous, instantaneous power and frequency evolve separately from each  
% other, according to an (aperiodic) autoregressive processes; they also evolve
% separately for each channel. In the current version there is only one
% fundamental frequency (i.e. each time point has one single instantaneous
% frequency value). The dynamics of these processes depend on the
% parameters POW_AR_W (for power) and FREQ_AR_W (for frequency); the closer
% to 1.0 these are the slower are their dynamics. The parameters POW_RANGE
% and FREQ_RANGE mark the interval within which each is allowed to vary
% (out of this interval, it saturates). These parameters are within
% the spont_options struct parameter of the function.
%
% The stimulus-evoked activity is parametrised in evoked_options, 
% and can be of four types:
%
% - Phase modulation, where the phase resets to a certain phase given by
% the parameter PH, which by default it randomly takes the values -pi/2 or
% pi/2. There can be one value for each class of stimulus (Q x 1 vector), or
% one value for each class of stimulus and channel (Q x nchan matrix). PH
% can be manually specified but it is recommended to contain only -pi/2
% and +pi/2 values (the trough and crest of the oscillation, respectively)
%
% - Power modulation, which currently is a multiplication factor (given by POW)
% to the current instantaneous power signal. It could easily be modified to be 
% one absolute value of power per stimulus, which would probably be easier
% to decode. POW can be a (Q x 1) vector or a (Q x nchan) matrix, where Q
% is the number of different stimuli.
%
% - Frequency modulation, which currently is a multiplication factor (given by FREQ)
% to the current instantaneous frequency. It could easily be modified to be 
% one absolute value of frequency per stimulus, which would probably be easier
% to decode. FREQ can be a (Q x 1) vector or a (Q x nchan) matrix.
%
% - Signal injection, which is just an addition of a given value to the
% signal, looking like a bump. This additive value is given by parameter
% INJ, which can be a (Q x 1) vector or a (Q x nchan) matrix.
%
% There can be any combination of these effects; which ones we get can
% be specified with the boolean parameters phase_reset, power_modulation,
%  frequency_modulation and signal_inject. 
% 
% Additionally to PH, POW, FREQ and INJ, there are also parameters DELAY_PH,
% DELAY_POW, DELAY_FREQ and DELAY_INJ, which introduce different effect delays
% for the different classes of stimulus. These make a different type of
% effect that is purely temporal. All of these should be (Q x nchan) matrices
% Unstructured temporal jitters (non-stimulus specific) are specified by parameters
% DELAY_PH_JITTER, DELAY_POW_JITTER, DELAY_FREQ_JITTER and DELAY_INJ_JITTER
%
% The temporal window during which these effects exert their influence is given by a 
% kernel function, and is different for each type of effect. Currently, this is
% hard-coded in the function convolveStimulus.m. This is not stimulus or
% channel specific (only specific to the type of effect), so it's not
% "decodable" 
%
% Overall, there are all possible combinations of the following
% options for the stimulus effect:
% - phase modulation vs frequency modulation vs amplitude modulation vs signal injection
% - delay effect (DELAY_PH,DELAY_POW,DELAY_INJ) vs distinct modulation (PH,POW,INJ)
% - channel-absolute vs channel-relative (whether these parameters have one value
%   per stimulus class and channel, or common values for all channels)
%
% As it stands, the default options are: both delay and distinct-modulation 
% effects occur. For the delay the effect is relative, whereas the properties of the effect
% (eg which phase channels reset to) are absolute (same for all channels).
%
% Finally, the data is projected into a higher (but low-rank) space, trying
% to simulate a reduced number of sources in the brain projected into a
% large number of sensors. 

% Diego Vidaurre, Aarhus University (2021)

% spontaneous activity options 
if nargin < 3 || isempty(spont_options)
    spont_options = struct();
end
if ~isfield(spont_options,'POW_AR_W'), POW_AR_W = 0.9999; 
else, POW_AR_W = spont_options.POW_AR_W;
end
if ~isfield(spont_options,'FREQ_AR_W'), FREQ_AR_W = 0.9995; 
else, FREQ_AR_W = spont_options.FREQ_AR_W;
end
if ~isfield(spont_options,'POW_RANGE'), POW_RANGE = [0.1 3];
else, POW_RANGE = spont_options.POW_RANGE;
end
if ~isfield(spont_options,'FREQ_RANGE'), FREQ_RANGE = [0.0001 0.1]; 
else, FREQ_RANGE = spont_options.FREQ_RANGE;
end
if ~isfield(spont_options,'NOISE'), NOISE = 0.5; 
else, NOISE = spont_options.NOISE;
end

% evoked activity options 
if nargin < 4 || isempty(evoked_options)
    evoked_options = struct(); 
    evoked_options.phase_reset = 0;
    evoked_options.power_modulation = 0;
    evoked_options.frequency_modulation = 0;
    evoked_options.signal_inject = 0;
else
    if ~isfield(evoked_options,'phase_reset')
        evoked_options.phase_reset = 0;
    end
    if ~isfield(evoked_options,'frequency_modulation')
        evoked_options.frequency_modulation = 0;
    end
    if ~isfield(evoked_options,'power_modulation')
        evoked_options.power_modulation = 0;
    end
    if ~isfield(evoked_options,'signal_inject')
        evoked_options.signal_inject = 0;
    end
end
if isfield(evoked_options,'stimulus') 
    stim = evoked_options.stimulus; resting = 0; 
    Q = length(unique(stim))-1; 
    if size(stim,1) ~= sum(T)
        error('Incorrect stimulus dimension')
    end
else
    resting = 1; 
    if evoked_options.phase_reset || evoked_options.signal_inject ...
            || evoked_options.power_modulation || evoked_options.frequency_modulation
        disp('No stimulus was specified.')
        evoked_options.phase_reset = 0; 
        evoked_options.signal_inject = 0;
        evoked_options.power_modulation = 0;
        evoked_options.frequency_modulation = 0;
    end
    disp('Pure resting state data')
end
% Phase modulation. 
% Note that a phase reset is not really a "reorganisation of activity"
% in the sense that it doesn't depend on ongoing patterns (it just resets)
if evoked_options.phase_reset 
    % delay of the response, how long it takes for the brain to react
    if ~isfield(evoked_options,'DELAY_PH'), DELAY_PH = 50 + 5 * rand(Q,nchan);
    else
        DELAY_PH = evoked_options.DELAY_PH;
        if length(DELAY_PH(:))==Q, DELAY_PH = repmat(DELAY_PH,1,nchan); end
    end % if all the columns are equal is an absolute modulation, otherwise relative
    DELAY_PH(DELAY_PH<1) = 1;
    if ~isfield(evoked_options,'DELAY_PH_JITTER'), DELAY_PH_JITTER = 2.5;
    else, DELAY_PH_JITTER = evoked_options.DELAY_PH_JITTER;
    end 
    if ~isfield(evoked_options,'PH'), PH = 2*rand(Q,1)-1; PH(PH<0)=-pi/2; PH(PH>=0)=pi/2; 
        PH = repmat(PH,1,nchan);
    else
        PH = evoked_options.PH; % within [-1,1]; disregards whether the signal goes up or down
        if length(PH(:))==Q, PH = repmat(PH,1,nchan); end
    end % if all the columns are equal is an absolute modulation, otherwise relative    
end 
% Power modulation, unrelated to the ongoing phase
if evoked_options.power_modulation
    % delay of the response, how long it takes for the brain to react
    if ~isfield(evoked_options,'DELAY_POW'), DELAY_POW = 50 + 5 * rand(Q,nchan);
    else, DELAY_POW = evoked_options.DELAY_POW;
    end
    if ~isfield(evoked_options,'DELAY_POW_JITTER'), DELAY_POW_JITTER = 2.5;
    else, DELAY_POW_JITTER = evoked_options.DELAY_POW_JITTER;
    end % if all the columns are equal is an absolute modulation, otherwise relative
    if ~isfield(evoked_options,'POW'), POW = 2*rand(Q,1)+1; POW = repmat(POW,1,nchan);
    else
        POW = evoked_options.POW;
        if length(POW(:))==Q, POW = repmat(POW,1,nchan); end
    end % if all the columns are equal is an absolute modulation, otherwise relative
end
% Frequency modulation, unrelated to the ongoing phase
if evoked_options.frequency_modulation
    % delay of the response, how long it takes for the brain to react
    if ~isfield(evoked_options,'DELAY_FREQ'), DELAY_FREQ = 50 + 5 * rand(Q,nchan);
    else, DELAY_FREQ = evoked_options.DELAY_FREQ;
    end
    if ~isfield(evoked_options,'DELAY_FREQ_JITTER'), DELAY_FREQ_JITTER = 2.5;
    else, DELAY_FREQ_JITTER = evoked_options.DELAY_FREQ_JITTER;
    end % if all the columns are equal is an absolute modulation, otherwise relative
    if ~isfield(evoked_options,'FREQ'), FREQ = 1.5*rand(Q,1)+0.5; FREQ = repmat(FREQ,1,nchan);
    else
        FREQ = evoked_options.FREQ;
        if length(FREQ(:))==Q, FREQ = repmat(FREQ,1,nchan); end
    end % if all the columns are equal is an absolute modulation, otherwise relative
end
% Additive component in the signal
if evoked_options.signal_inject
    % delay of the response, how long it takes for the brain to react
    if ~isfield(evoked_options,'DELAY_INJ'), DELAY_INJ = 50 + 5 * rand(Q,nchan);
    else, DELAY_INJ = evoked_options.DELAY_INJ;
    end
    if ~isfield(evoked_options,'DELAY_INJ_JITTER'), DELAY_INJ_JITTER = 2.5;
    else, DELAY_INJ_JITTER = evoked_options.DELAY_INJ_JITTER;
    end % if all the columns are equal is an absolute modulation, otherwise relative
    if ~isfield(evoked_options,'INJ'), INJ = 2*randn(Q,1); INJ = repmat(INJ,1,nchan);
    else
        INJ = evoked_options.INJ;
        if length(INJ(:))==Q, INJ = repmat(INJ,1,nchan); end
    end % if all the columns are equal is an absolute modulation, otherwise relative     
end 
REFRACTORY = 50; EPS = 1e-2;



if nargin < 5
    head_model = [];
elseif length(head_model) == 1 % number of correlated "sensors"
    head_model = rand(nchan,head_model);
end
if nargin < 6
    show = 0;
end
    
N = length(T);

X = cell(N,1);
Freq = cell(N,1);
Power = cell(N,1);

for j = 1:N
    
    %%% Generating the spontaneous dynamics
    f = randn(T(j)+100,nchan); % derivative of phase (instantaneous freq)
    for c = 1:nchan
        for t = 2:T(j)+100, f(t,c) = f(t-1,c) * FREQ_AR_W + f(t,c); end
        f(:,c) = f(:,c) - min(f(:,c));
        f(:,c) = f(:,c) / max(f(:,c)) * (FREQ_RANGE(2)-FREQ_RANGE(1)) + FREQ_RANGE(1);
    end
    f = f(101:end,:);
    
    p = randn(T(j)+100,nchan); % amplitude
    for c = 1:nchan
        for t = 2:T(j)+100, p(t,c) = p(t-1,c) * POW_AR_W + p(t,c); end
        p(:,c) = p(:,c) - min(p(:,c));
        p(:,c) = p(:,c) / max(p(:,c)) * (POW_RANGE(2)-POW_RANGE(1)) + POW_RANGE(1);
    end
    p = p(101:end,:);
        
    x = zeros(T(j),nchan); % signal
    
    if resting
        for c = 1:nchan
            x(:,c) = sin(cumsum(f(:,c))) .* p(:,c) + NOISE * randn(T(j),1);
        end
    else %%% Generating the task-evoked effects
        cstim = zeros(T(j),Q,nchan); ind = (1:T(j))+sum(T(1:j-1));
        
        % Frequency effect
        if evoked_options.frequency_modulation
            for c = 1:nchan % get convolved stimulus
                for k = 1:Q
                    if length(DELAY_FREQ_JITTER(:))==1, jit = DELAY_FREQ_JITTER;
                    else, jit = DELAY_FREQ_JITTER(k,c);
                    end
                    cstim(:,k,c) = convolveStimulus(stim(ind)==k,'frequency',DELAY_FREQ(k,c),jit);
                end
            end
            for t = 1:T(j)
                for c = 1:nchan
                    for k = 1:Q
                        alpha = cstim(t,k,c); % weight for stimulus response
                        if alpha>0
                            rho = 1 + alpha * (FREQ(k,c) - 1); % >1 multiplier
                            f(t,c) = rho * f(t,c);
                        end
                    end
                end
            end
        end
        
        % phase effect
        if evoked_options.phase_reset
            for c = 1:nchan % get convolved stimulus
                for k = 1:Q
                    if length(DELAY_PH_JITTER(:))==1, jit = DELAY_PH_JITTER;
                    else, jit = DELAY_PH_JITTER(k,c);
                    end
                    cstim(:,k,c) = convolveStimulus(stim(ind)==k,'phase',DELAY_PH(k,c),jit);
                end
            end
            x(1,:) = (2*pi) * rand(1,nchan) - pi; % initial random phase (radians)
            wait = zeros(Q,1); % refractory period: when we touch the targetted phase, 
            % stimulus loses influence for a bit
            for t = 2:T(j)
                x(t,:) = x(t-1,:); 
                for c = 1:nchan
                    x(t,c) = x(t,c) + f(t,c);           
                    for k = 1:Q
                        alpha = cstim(t,k,c); % weight for stimulus response
                        if alpha>0 && ~wait(k)
                            delta = polarGradient(x(t,c),PH(k,c));
                            x(t,c) = x(t,c) + alpha * delta; 
                            if abs(x(t,c)-PH(k,c))<EPS, wait(k) = REFRACTORY; end
                        elseif wait(k)
                            wait(k) = wait(k) - 1;
                        end % the gradient can overshoot if there are overlapping stim
                    end
                    if x(t,c) >= pi, x(t,c) = x(t,c) - 2*pi; end
                    if x(t,c) < -pi, x(t,c) = x(t,c) + 2*pi; end
                end
            end
        else
            for c = 1:nchan
                x(:,c) = cumsum(f(:,c));
            end
        end
        
        % transform phase into signal value
        x = sin(x); 
                
        % amplitude effect
        if evoked_options.power_modulation
            for c = 1:nchan % get convolved stimulus
                for k = 1:Q
                    if length(DELAY_POW_JITTER(:))==1, jit = DELAY_POW_JITTER;
                    else, jit = DELAY_POW_JITTER(k,c);
                    end
                    cstim(:,k,c) = convolveStimulus(stim(ind)==k,'power',DELAY_POW(k,c),jit);
                end
            end
            for t = 1:T(j)
                for c = 1:nchan
                    for k = 1:Q
                        alpha = cstim(t,k,c); % weight for stimulus response
                        if alpha>0
                            rho = 1 + alpha * (POW(k,c) - 1); % >1 multiplier
                            p(t,c) = rho * p(t,c);
                        end
                    end
                end
            end
        end
        
        % scale signal by power
        x = x .* p;
        
        % signal injection
        if evoked_options.signal_inject
            for c = 1:nchan % get convolved stimulus
                for k = 1:Q
                    if length(DELAY_INJ_JITTER(:))==1, jit = DELAY_INJ_JITTER;
                    else, jit = DELAY_INJ_JITTER(k,c);
                    end
                    cstim(:,k,c) = convolveStimulus(stim(ind)==k,'signal_inj',DELAY_INJ(k,c),jit);
                end
            end
            for t = 1:T(j)
                for c = 1:nchan
                    for k = 1:Q
                        alpha = cstim(t,k,c); % weight for stimulus response
                        if alpha>0
                            x(t,c) = x(t,c) + alpha * INJ(k,c);
                        end
                    end
                end
            end
        end
        
        % additive gaussian noise
        x = x + NOISE * randn(T(j),nchan);
    end
    
    Freq{j} = f;
    Power{j} = p;
    X{j} = x;
    
    if ~isempty(head_model)
        x = x * head_model; 
    end
    
    %%% plot if asked
    if show && j==1
        figure;
        subplot(411);
        plot(f(:,1));title('Instantaneous frequency')
        subplot(412);
        plot(p(:,1));title('Instantaneous power')
        subplot(413);
        plot(sin(cumsum(f(:,1))));ylim([-1.25 1.25]);title('Signal(f)')
        subplot(414);
        plot(x(:,1));title('Signal')
        figure;
        subplot(121); hist(f(:,1),100); title('Instantaneous frequency')
        subplot(122); hist(p(:,1),100); title('Instantaneous power')
    end
    
end

X = cell2mat(X); 
Freq = cell2mat(Freq); 
Power = cell2mat(Power); 

% reorder according to correlation 
m = pca(X,'NumComponents',1); [~,m] = sort(m);
X = X(:,m);
Freq = Freq(:,m);
Power = Power(:,m);

end


function delta = polarGradient(rho,rho_target)
d = rho_target - rho;
if abs(d) <= pi
    delta = d;
elseif rho < 0
    rho_target = rho_target - 2*pi;
    delta = rho_target - rho;
else
    rho_target = rho_target + 2*pi;
    delta = rho_target - rho;
end
end
