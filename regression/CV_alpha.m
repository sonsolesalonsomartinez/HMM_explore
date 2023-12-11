function best_alpha = CV_alpha(X,Y,alpha,responses,NCV)

[N,p] = size(X); 
NA = length(alpha);
intercept = all(X(:,1)==1);

if nargin < 5, NCV = 10; end
if nargin<4 || isempty(responses)
    c2 = cvpartition(N,'KFold',NCV);
else
    c2 = cvpartition(responses,'KFold',NCV);
end
c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = find(c2.training(icv));
    c.test{icv} = find(c2.test(icv));
end; clear c2

accuracy = zeros(NA,NCV);
for icv = 1:NCV
    Xtrain = X(c.training{icv},:);
    Ytrain = Y(c.training{icv});
    Xtest = X(c.test{icv},:);
    Ytest = Y(c.test{icv});
    C1 = Xtrain' * Xtrain;
    C2 = Xtrain' * Ytrain;
    for ialpha = 1:NA
        R = alpha(ialpha) * eye(p);
        if intercept, R(1,1) = 0; end
        b = (C1 + R) \ C2;
        Yhat = Xtest * b;
        if length(unique(Y(:)))==2
            accuracy(ialpha,icv) = mean(sign(Yhat)==sign(Ytest));
        else % here the minimum should be pushed forward. 
            accuracy(ialpha,icv) = mean(abs(Yhat-Ytest));
        end
        % in the paper it was the second one
        
    end
end

accuracy = mean(accuracy,2);
if length(unique(Y(:)))==2
    [~,I] = max(accuracy);
else
    [~,I] = min(accuracy);
end    
best_alpha = alpha(I);

end

