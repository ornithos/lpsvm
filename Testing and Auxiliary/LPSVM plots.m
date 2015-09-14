[grdS,grdD] = meshgrid([5,10,20], [0.1,0.2]);

[grdS,grdD] = meshgrid([10,20,60], [10, 1,0.1]);
nSV = zeros(3,3);
testperf = zeros(3,3);
timesv = zeros(3,3);
for i = 1:numel(grdS)
    fprintf('%d',i);
    tstt = tic;
    cl = fitcsvm(trainSet',trainLbl','KernelFunction','rbf',...
        'KernelScale',grdS(i),'BoxConstraint',grdD(i),'ClassNames',[-1,1]);
    timesv(i) = toc(tstt);
    fittedVals = predict(cl,testSet');
    testperf(i) = sum(fittedVals==testLbl')/(numel(fittedVals));
    nSV(i) = sum(cl.IsSupportVector);
end


for i = 1:numel(grdS)
    clear kernelCache
    sigma  = grdS(i);
    kernel = @(x, y)gaussK(x, y, sigma);
    D      = grdD(i);

    %% Train and test model
    % Call the model
    rng('default');
    fprintf('Beginning training...%d\n',i);
    model = lpsvm(trainSet, trainLbl, D, kernel, initm, maxiter, ...
        tol, 0, sinit, 0);

    fittedVals = predictLPSVM(model, trainSet);
    fprintf('Training Error: %2.4f%%\n', ...
        sum(trainLbl ~= fittedVals)*100.0/length(trainLbl));
end
% model = lpsvm_antibounce(trainSet, trainLbl, 0.2, kernel, initm, maxiter, ...
%     tol, debug, sinit, 0);
sigma = 20;
D = 0.1;
kernel = @(x, y)gaussK(x, y, sigma);
clear kernelCache
model = lpsvm(trainSet, trainLbl, D, kernel, initm, maxiter, ...
        tol, 0, sinit, 0);
clear kernelCache
model500 = lpsvmAD(trainSet, trainLbl, D, kernel, initm, maxiter, ...
        tol, 0, sinit, 0);
    clear kernelCache
model1000 = lpsvmAD(trainSet, trainLbl, D, kernel, initm, maxiter, ...
        tol, 0, sinit, 0);
    clear kernelCache
model250 = lpsvmAD(trainSet, trainLbl, D, kernel, initm, maxiter, ...
        tol, 0, sinit, 0);


% Calculate actual weights of SVs for prediction
wgttol = 1e-5;
[dim, N] = size(trainSet);
[~ , tN] = size(testSet);

svmmod = model.model;
nModels = length(svmmod);
alpha = zeros(nModels,N);

% recover alpha weights...
% weights by iter
for i = 1:nModels
    idx = svmmod{i}.i;
    alpha(i,idx) = (svmmod{i}.u).*trainLbl(idx)./svmmod{i}.v;
end

% cum weights after applying a
% case for iter 1 handled trivially since a = 1 by definition.
a = zeros(nModels,nModels);
a(1,1) = 1;
for i = nModels:(-1):2
    a(i,1:i) = model.aMem(1:i, i-1);
    a(i,a(i,:) < wgttol) = 0;
    
    alpha(i,:) = a(i,:)*alpha;
end

trainScore = zeros(1, nModels);
allModels = cell(1, nModels);

% set up test model and score
if tN < 1000
    error('test Set has fewer than 100 columns - is this the transpose?');
end
chunks = chunkVector(tN, 0.5*10^8/length(model.Alpha));
testScore  = zeros(numel(chunks), nModels);

for cC = 1:numel(chunks)
    fprintf('test round %d of %d\n',cC, numel(chunks));
    testSettmp = testSet(:,chunks{cC});
    testLbltmp = testLbl(chunks{cC});
    
    for cM = 1:nModels
        tst = fprintf('%d ', cM);
        if mod(cM,30) == 0; fprintf('\n'); end
        [nSVs, SVset] = countSV(svmmod, cM, a(cM, 1:cM));
        testmod.IsSupportVector = zeros(N,1);
        svnum = 1;
        for i = 1:N
            if i == SVset(svnum)
                testmod.IsSupportVector(i) = 1;
                svnum = svnum + 1;
            end
            if(svnum > nSVs); break; end
        end

        testmod.IsSupportVector = logical(testmod.IsSupportVector);
        testmod.Alpha = alpha(cM, testmod.IsSupportVector);
        testmod.SVs = trainSet(:, testmod.IsSupportVector);
%         testmod.SVs = trainSet(testmod.IsSupportVector,:)';
        testmod.kernel = kernel;
%         allModels{cM} = testmod;
        SVs(cM) = sum(testmod.IsSupportVector);

        testScore(cC,cM) = sum(testLbltmp ~= predictLPSVM(testmod, testSettmp, false))/(1.0*tN);
    end
    clear kernelCache
    beep
end
testScoreTot = sum(testScore*tN)./tN;

% fprintf('training error:...\n');
% clear kernelCache
% for cM = 1:nModels
%     testmod = allModels{cM};
%     trainScore(cM) = sum(trainLbl ~= predictLPSVM(testmod, trainSet))/(1.0*N);
% end

obj = model.betahist;

figure
plot(1:length(testScore), testScore, 'b*-')
figure
plot(1:length(trainScore), trainScore, 'g*-')
    
    
SVs = zeros(1,nModels);
for i = 1:nModels
    SVs(i) = length(allModels{i}.Alpha);
end
SVs = sum(abs(alpha)>0,2)';
    
figure;
semilogy(SVs, testScoreTot, 'r-o');
title('Support Vectors vs Test Error')
xlabel('Support Vectors');
ylabel('Error');
hold on;
%plot(2575,1-0.9774,'k*','MarkerSize',10);  % 10k subsample
plot(29609,1-0.8972,'k*','MarkerSize',10);  % full 60k
legend('LPSVM','LIBSVM')


figure;
plot(1:length(model.time), model.time, 1:length(model.time), ...
    model.time_kern(2:151), 'LineWidth', 2)
title('Time per Epoch')
xlabel('Epochs');
ylabel('Time (s)');
legend('Solver','Kernel');

libOptions =  sprintf('%s %s','-s 1 -t 2 -g', ...
    num2str(1/30), ' -n', num2str(0.0005));
tic;libModel = svmtrain(trainLbl', trainSet', libOptions);toc;
[pred, accuracy, dec] = svmpredict(testLbl', testSet', libModel);


% results matrix
M = zeros(5,5);
iterRng = [2,52,102,152,202];
M(2:5,1) = [sum(model.time_kern(1:2)); 0; SVs(2); testScore(2)];
for i = 2:5
    M(2:5,i) = [sum(model.time_kern(1:iterRng(i))); 
            sum(model.time(1:(iterRng(i)-2))); SVs(iterRng(i)); 
            testScore(iterRng(i))];
end

% for Venn Diagram
% benchmark nearest points on SV curve for D=1,0.1 with first 14 of D=0.01
numP = 15;
crossover = zeros(numP,7);
SVmatches = [zeros(numP,2) MNIST9S20.SVs(1:numP,3) ];
for cI = 1:numP
    [~,SVmatches(cI,1)] = min(abs(MNIST9S20.SVs(:,1) - SVmatches(cI,3)));
    [~,SVmatches(cI,2)] = min(abs(MNIST9S20.SVs(:,2) - SVmatches(cI,3)));
end
SVmatches(:,3) = 1:numP;

% Calculate crossover of 3-way venn diagram
cSVs = cell(3,1);
for cI = 1:numP
    cSVs{1} = find(gameOn{1,1}{SVmatches(cI,1)}.IsSupportVector);
    cSVs{2} = find(gameOn{2,1}{SVmatches(cI,2)}.IsSupportVector);
    cSVs{3} = find(gameOn{3,1}{SVmatches(cI,3)}.IsSupportVector);
    cross12 = ismember(cSVs{1}, cSVs{2});
    cross13 = ismember(cSVs{1}, cSVs{3});
    cross23 = ismember(cSVs{2}, cSVs{3});
    cross123 = ismember(cSVs{1}(cross12), cSVs{3});
    crossover(cI,1) = length(cSVs{1}) - sum(cross12) - sum(cross13) + sum(cross123);
    crossover(cI,2) = length(cSVs{2}) - sum(cross23) - sum(cross12) + sum(cross123);
    crossover(cI,3) = length(cSVs{3}) - sum(cross13) - sum(cross23) + sum(cross123);
    crossover(cI,4) = sum(cross12) - sum(cross123);
    crossover(cI,5) = sum(cross23) - sum(cross123);
    crossover(cI,6) = sum(cross13) - sum(cross123);
    crossover(cI,7) = sum(cross123);
end

% return vector for each D and each iter denoting the free and maxed SVs
allD = [1,0.1,0.01];
typeSV = cell(60, 3);
for cJ = 1:3
    for cI = 3:62
        typeSV{cI,cJ} = gameOn{cJ,2}.model{cI}.u;
    end
end

% Calculate crossover for free SVs
cSVs = cell(3,1);
for cI = 3:numP
    cSVs{1} = find(gameOn{1,1}{SVmatches(cI,1)}.IsSupportVector);
    cSVs{2} = find(gameOn{2,1}{SVmatches(cI,2)}.IsSupportVector);
    cSVs{3} = find(gameOn{3,1}{SVmatches(cI,3)}.IsSupportVector);
    cross12 = ismember(cSVs{1}, cSVs{2});
    cross13 = ismember(cSVs{1}, cSVs{3});
    cross23 = ismember(cSVs{2}, cSVs{3});
    cross123 = ismember(cSVs{1}(cross12), cSVs{3});
    crossover(cI,1) = sum(typeSV{cI,1}(cSVs{1})) - sum(cross12) - sum(cross13) + sum(cross123);
    crossover(cI,2) = length(cSVs{2}) - sum(cross23) - sum(cross12) + sum(cross123);
    crossover(cI,3) = length(cSVs{3}) - sum(cross13) - sum(cross23) + sum(cross123);
    crossover(cI,4) = sum(cross12) - sum(cross123);
    crossover(cI,5) = sum(cross23) - sum(cross123);
    crossover(cI,6) = sum(cross13) - sum(cross123);
    crossover(cI,7) = sum(cross123);
end
% Calculate crossover for violating SVs

if false

    %%%%%%%%%%%%%%%%%%%%
    % load MNIST9S20_iter_experiment.mat

    % Training Error vs Iterations
    figure;
    plot(MNIST9S20.iters(:,1), MNIST9S20.trainScore(:,1), 'b*-', ...
        MNIST9S20.iters(:,2), MNIST9S20.trainScore(:,2), 'g*-', ...
        MNIST9S20.iters(:,3), MNIST9S20.trainScore(:,3), 'r*-');
    title('Iterations vs Training Error')
    xlabel('Iteration');
    ylabel('Error');
    legend('D = 1','D = 0.1','D = 0.01');

    % Test Error vs Iterations
    figure;
    plot(MNIST9S20.iters(:,1), MNIST9S20.testScore(:,1), 'b*-', ...
        MNIST9S20.iters(:,2), MNIST9S20.testScore(:,2), 'g*-', ...
        MNIST9S20.iters(:,3), MNIST9S20.testScore(:,3), 'r*-');
    title('Iterations vs Test Error')
    xlabel('Iteration');
    ylabel('Error');
    legend('D = 1','D = 0.1','D = 0.01');

    % Training Error vs SVs
    figure;
    plot(MNIST9S20.SVs(:,1), MNIST9S20.trainScore(:,1), 'b*-', ...
        MNIST9S20.SVs(:,2), MNIST9S20.trainScore(:,2), 'g*-', ...
        MNIST9S20.SVs(:,3), MNIST9S20.trainScore(:,3), 'r*-');
    title('Support Vectors vs Training Error')
    xlabel('Support Vectors');
    ylabel('Error');
    legend('D = 1','D = 0.1','D = 0.01');

    % Training Error vs SVs (Truncated)
    figure;
    plot(MNIST9S20.SVs(1:53,1), MNIST9S20.trainScore(1:53,1), 'b*-', ...
        MNIST9S20.SVs(1:45,2), MNIST9S20.trainScore(1:45,2), 'g*-', ...
        MNIST9S20.SVs(1:15,3), MNIST9S20.trainScore(1:15,3), 'r*-');
    title('Support Vectors vs Training Error')
    xlabel('Support Vectors');
    ylabel('Error');
    legend('D = 1','D = 0.1','D = 0.01');

    % Test Error vs SVs
    figure;
    plot(MNIST9S20.SVs(:,1), MNIST9S20.testScore(:,1), 'b*-', ...
        MNIST9S20.SVs(:,2), MNIST9S20.testScore(:,2), 'g*-', ...
        MNIST9S20.SVs(:,3), MNIST9S20.testScore(:,3), 'r*-');
    title('Support Vectors vs Test Error')
    xlabel('Support Vectors');
    ylabel('Error');
    legend('D = 1','D = 0.1','D = 0.01');

    % Test Error vs SVs (Truncated)
    figure;
    plot(MNIST9S20.SVs(:,1), MNIST9S20.testScore(:,1), 'b*-', ...
        MNIST9S20.SVs(:,2), MNIST9S20.testScore(:,2), 'g*-', ...
        MNIST9S20.SVs(1:19,3), MNIST9S20.testScore(1:19,3), 'r*-');
    title('Support Vectors vs Test Error')
    xlabel('Support Vectors');
    ylabel('Error');
    legend('D = 1','D = 0.1','D = 0.01');


    % SVs by Iteration
    figure;
    plot(MNIST9S20.iters(:,1), MNIST9S20.SVs(:,1), 'b*-', ...
        MNIST9S20.iters(:,2), MNIST9S20.SVs(:,2), 'g*-', ...
        MNIST9S20.iters(:,3), MNIST9S20.SVs(:,3), 'r*-');
    title('Support Vectors by Iteration')
    xlabel('Iteration');
    ylabel('Support Vectors');
    legend('D = 1','D = 0.1','D = 0.01', 'Location', 'northwest');


    % SVs by Iteration
    figure;
    plot(MNIST9S20.iters(1:(end-1),1), diff(MNIST9S20.SVs(:,1)), 'b*-', ...
        MNIST9S20.iters(1:(end-1),2), diff(MNIST9S20.SVs(:,2)), 'g*-', ...
        MNIST9S20.iters(1:(end-1),3), diff(MNIST9S20.SVs(:,3)), 'r*-');
    title('Additional Support Vectors by Iteration')
    xlabel('Iteration');
    ylabel('Support Vectors');
    legend('D = 1','D = 0.1','D = 0.01');
end




SVall = false(50000,152);
for cM = 1:nModels
        tst = fprintf('%d ', cM);
        if mod(cM,30) == 0; fprintf('\n'); end
        [nSVs, SVset] = countSV(svmmod, cM, a(cM, 1:cM));
        SVall(SVset,cM) = true;
end
    
for cC = 1:numel(chunks)
    fprintf('test round %d of %d\n',cC, numel(chunks));
    testSettmp = testSet(:,chunks{cC});
    testLbltmp = testLbl(chunks{cC});
    
    for cM = 1:nModels
        tst = fprintf('%d ', cM);
        if mod(cM,30) == 0; fprintf('\n'); end
        
        testmod.IsSupportVector = SVall(:,cM);
        testmod.Alpha = alpha(cM, testmod.IsSupportVector);
        testmod.SVs = trainSet(:, testmod.IsSupportVector);
        testmod.kernel = kernel;
        
        testScore(cC,cM) = sum(testLbltmp ~= predictLPSVM(testmod, testSettmp, false))/(1.0*tN);
    end
    clear kernelCache
    beep
end
testScoreTot = sum(testScore*tN)./tN;