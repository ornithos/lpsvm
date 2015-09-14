%% Test LPSVM using Toy Data
rng('default');
addpath '../3. C++ Extensions'
m = 400;  % num examples in each class
tst_pct = 0.3; % pct data used to test model

g1 = repmat([1,1,1],m,1) + randn(m,3)*chol([2,1,0.5;1,1.5,0.75;0.5,0.75,1]);
g2 = repmat([2,-1,0],m,1) + randn(m,3)*chol([2,0.5,0.2;0.5,1.5,1.25;0.2,1.25,2]);

% Use only 2D Gaussians so we can visualise
g1 = [ones(size(g1,1),1) g1(:,1:2)];
g2 = [ones(size(g2,1),1) g2(:,1:2)];

% split into training and test set
split = randperm(m*2);
split = {split(1:(m*2*tst_pct)), split((m*2*tst_pct+1):end)};

trainSet = [g1; g2];
testSet = trainSet(split{1},:)';
trainSet = trainSet(split{2},:)';

trainLbl = [ones(m,1); -ones(m,1)];
testLbl = trainLbl(split{1})';
trainLbl = trainLbl(split{2})';


%% Set parameters
% Kernel
sigma = 2;
kernel = @(x, y)gaussK(x, y, sigma);

%% Model paramterss
initm   = 2;   % Initial number of models
maxiter = 10;  % Maximum iterations
nu      = 0.05; % nu (minimum frac of support vector / % noise)
D       = 1/(nu*2*m*(1-tst_pct)); % Cost weight hyperparameter
D       = 0.03;  % IGNORING NU FOR NOW - so can interpret how algm scales
debug   = 1;
tol     = .2e-3;
sinit   = 0.05;

%% Train and test model
% Call the model

model = lpsvm(trainSet, trainLbl, D, kernel, initm, maxiter, ...
    tol, debug, sinit, 0);   % 1 = remove rows where a == 0
% model = lpsvm_antibounce(trainSet, trainLbl, [0.05 0.05], kernel, initm, ...
%     maxiter, tol, debug, sinit, 0);

figure;
plot(model.time, 'g*-')
title('Time per LP')

figure;
plot(-model.betahist, 'b*-')
hold on
plot(-model.vjhist, 'r*-');


sparsity = zeros(1, length(model.model));

for i=1:length(model.model),
    sparsity(i) = size(model.model{i}.x, 2);
end

figure
hist(sparsity);
fittedVals = predictLPSVM(model, trainSet);
fprintf('Training Error: %2.4f%%\n', ...
    sum(trainLbl ~= fittedVals)*100.0/length(trainLbl));
kernelCache(-1);
fprintf('Test Error:\n');
fittedVals = predictLPSVM(model, testSet);
testErr = sum(testLbl ~= fittedVals)/(1.0*size(testSet,2));
fprintf(' %1.4f\n', testErr);
confusionmat(testLbl, fittedVals)

if m < 3000; PlotLPSVM(model, trainSet, trainLbl, false); end