function fail = activeSetTesting(H, D, options, lastActive, debug, p, u)
% activeSetTesting
% Test number of iterations required to convergence for various activeSet
% strategies

Nvals = 10:10:200;
l = numel(Nvals);
resIter = zeros(1,l);
resAddl = zeros(1,l);


for ii = 1:l
    NN = Nvals(ii);    
    % Top NN
    [it, b, sz, nz] = solveLPActiveABtest(H, D, options, lastActive, NN, 0, debug);
    resIter(1,ii) = it;
    resAddl(1,ii) = (sz - nz)/(nz*1.0);
end

try
    results.NN = Nvals;
    results.iter = resIter;
    results.addl = resAddl;
    assignin('base',strcat('resultsAS',num2str(p)), results)
catch
    fail = 1;
end
end

