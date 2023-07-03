
% create signals with coherence. 

% this function is intended to generate signals that show coherence at some
% timepoints. It is done by mixing signals with some proportion. 
% The function at the moment takes as input the coherence parameter (time 
% series) and n+1 independent signals, and outputs n signals that show 
% coherence. 
% input signals should be of shape (time, n+1). the coherence parameter is
% expected of shape (time,1) and of values beteen 0 and 1 at all times.
% please make sure to have a smooth coherence parameter time series.

function output_signals = get_coherent_signals(input_signals, coherence_parameter)

n_signal = size(input_signals,2)-1;
n_timepoints = size(input_signals,1);
output_signals = NaN(n_timepoints, n_signal);

for i=1:n_signal
    output_signals(:,i) = (1-coherence_parameter) .* input_signals(:,i) + coherence_parameter .* input_signals(:,n_signal+1);
end


end
