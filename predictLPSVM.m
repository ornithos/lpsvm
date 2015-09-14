function out = predictLPSVM(model, test, bChunk)
% predictLPSVM
% Create predictions from the LPSVM model object
%   Arguments:
%   model      -   (LPSVM object) output of lpsvm function
%   test       -   (Matrix) test examples (columnwise)
%   bChunk     -   (Logical) Keep train/test kernel matrix in memory or
%                  split the test data into 'chunks' and clear kernel cache
%                  after approx 1 GB memory.
%                  This is not a good idea if one wishes to use the cached
%                  kernel values generated outside of this function.
%
%   Outputs:
%   out        -   Predicted labels corresponding to test examples.

    if ~exist('bChunk','var')
        bChunk = true;
    end
    
    % Admin
    maxmem = 1e9;  %1GB
    SVs    = find(model.IsSupportVector);
    nsv    = numel(SVs);
    [d, n] = size(test);
    if d ~= size(model.SVs,1)
        error('SVs are dimension %d, test data is dimension %d!', ...
            size(model.SVs,1), d);
    end
    
    if bChunk
        % Partition test set into manageable chunks
        maxColPerIter = floor(maxmem/(16*nsv));
        chunks        = chunkVector(n, maxColPerIter);
    else
        % or not...
        chunks = {1:n};
    end
    
    % preallocate prediction vector
    predict = zeros(1,n);
    
    % Perform test evaluation in chunks of < maxmem (default 1GB)
    for i = 1:numel(chunks)
        K = kkMat(test(:,chunks{i}), model.SVs, SVs, model.kernel, true);
        predict(chunks{i}) = model.Alpha*K;
        if bChunk
            clear kernelCache K
        end
    end
    
    % convert to binary classification
    out = sign(predict);
    out(out==0) = 1;     % hack!
end





