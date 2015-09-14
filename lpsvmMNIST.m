% Test of LPSVM for MNIST dataset

%% Add path
addpath('../2. David LPSVM/libsvm-3.20/matlab');  % libsvm
addpath('../3. C++ Extensions');

% Check whether we need to reload MNIST data
reloadMNIST = false;
if exist('trainSet', 'var')
    if size(trainSet, 1) ~= 784
        reloadMNIST = true;
    end
else
    reloadMNIST = true;
end
% reloadMNIST = true;

%% Reload MNIST (if applicable)
if reloadMNIST
    cd '../3. Data/MNIST'
    [trainSet,trainLbl,testSet,testLbl] = readMNIST(60000);  %(10000, 20000);
    cd '../../3. Code LPSVM Project'

    maxGrey = 255;
    % Change to double precision
    trainSet = double(trainSet)'/maxGrey;
    trainLbl = double(trainLbl)';
    testSet = double(testSet)'/maxGrey;
    testLbl = double(testLbl)';

    % Select problem (6 against rest)
    digit = [1,2,4,5,7];
    lblSet = 0:9;
    exampleIdx = ismember(trainLbl, lblSet);
    trainSet = trainSet(:,exampleIdx);
    trainLbl = trainLbl(exampleIdx);
    exampleIdx = ismember(testLbl, lblSet);
    testSet = testSet(:,exampleIdx);
    testLbl = testLbl(exampleIdx);

    trainLbl(~ismember(trainLbl,digit)) = -1;
    trainLbl(ismember(trainLbl,digit)) = 1;

    testLbl(~ismember(testLbl,digit)) = -1;
    testLbl(ismember(testLbl,digit)) = 1;
    m = length(trainLbl);
    
    trainSet = sparse(trainSet);
    testSet = sparse(testSet);
end


%% Set parameters
% Kernel
sigma = 30;
kernel = @(x, y)gaussK(x, y, sigma);

%% Model paramterss
initm   = 2;  % Initial number of models
maxiter = 20; % Maximum iterations
% nu      = 0.0001; % nu (minimum frac of support vector / % noise)
D       = 0.1; %1/(nu*m); % Cost weight hyperparameter
debug   = 1;
tol     = 1e-2;
sinit   = 1e-4;

%% Train and test model
% Call the model
rng('default');

% testerr = zeros(ii,kk);
% trainerr = zeros(ii,kk);

fprintf('Beginning training...\n');
lpsvmMtic = tic;
model = lpsvm(trainSet, trainLbl, D, kernel, initm, maxiter, ...
    tol, debug, sinit, 0, 0);
lpsvmMtoc = toc(lpsvmMtic);
lpsvmLtoc = sum(model.time);
fprintf('Time taken: %4.2f\n seconds', toc(lpsvmMtic));
% figure;
% plot(model.time, 'g*-')
% title('Time per LP')

figure;
plot(-model.betahist, 'b*-')
hold on
plot(-model.vjhist, 'r*-');
% plot(-model.lowb, 'r*-');
sparsity = zeros(1, length(model.model));

for i=1:length(model.model),
    sparsity(i) = size(model.model{i}.x, 2);
end
figure
hist(sparsity);

fittedVals = predictLPSVM(model, trainSet);
trainerr = sum(trainLbl ~= fittedVals)*100.0/length(trainLbl);
fprintf('time: %4.3f, Training Error: %2.4f%%\n', lpsvmMtoc, trainerr);

clear kernelCache
fprintf('Test Error:\n');
fittedVals = predictLPSVM(model, testSet);
testerr= sum(testLbl ~= fittedVals)/(1.0*size(testSet,2));
fprintf(' %1.4f\n', testerr);

confusionmat(testLbl, fittedVals)
