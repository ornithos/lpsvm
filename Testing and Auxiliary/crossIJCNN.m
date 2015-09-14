% Test of LPSVM for MNIST dataset

%% Add path
addpath('../2. David LPSVM/libsvm-3.20/matlab');  % libsvm
addpath('../3. C++ Extensions');

% Get data
[trainLbl, trainSet] = libsvmread('../3. Data/ijcnn01/ijcnn1.train');
[testLbl, testSet] = libsvmread('../3. Data/ijcnn01/ijcnn1.t');
m = length(trainLbl);

% Extract small cross-validation set to test parameters
rng('default');
splitIdx = randperm(m);
xvSize   = floor(0.8*m);
xvTrnSet = trainSet(splitIdx(1:xvSize),:);
xvTrnLbl = trainLbl(splitIdx(1:xvSize));


%% Reload MNIST (if applicable)
Cspan     = [0.05 0.01 0.005 0.001 ];
gammaSpan = [10 5 1];


acc = zeros(numel(Cspan),numel(gammaSpan));
for iC = 1:numel(Cspan),
    for iGamma = 1:numel(gammaSpan),
        C = Cspan(iC);
        gamma = gammaSpan(iGamma);
        
        libOptions =  sprintf('%s %s','-s 1 -t 2 -g', ...
                    num2str(gamma), ' -n', num2str(C), ' -v 5');
        
        tic;
        libModel = svmtrain(xvTrnLbl, xvTrnSet, libOptions);
        toc;
        %[pred, accuracy, dec] = svmpredict(testLbl, testSet, libModel);

        acc(iC,iGamma) = libModel;
        beep;
        pause(3)
    end
end

% BEST COMBINATION
C = 0.02;     % actually nu!
gamma = 2;