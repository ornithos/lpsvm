function [prediction, perModel] = predictLPSVMMulti(model, test, keepcache)
% predictLPSVM (model object, test data (columnwise))
% Create predictions from the LPSVM model object

    if exist('keepcache', 'var') && keepcache == false
        kernelCache(-1);   % Clear Cache from training run
    end
    if ~iscell(model)
        error('Model must be a cell array of models.')
    end
    
    nM = length(model);
    perModel = zeros(nM, size(test,2));
    for cI = 1:nM
        cModel = model{cI};
        K = kkMat(test, cModel.SVs, find(cModel.IsSupportVector), cModel.kernel, true);
        perModel(cI,:) = cModel.Alpha*K;
    end
    
    [~,prediction] = max(perModel, [], 1);
end





